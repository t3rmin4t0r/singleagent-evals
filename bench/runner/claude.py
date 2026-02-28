"""Claude runner — Anthropic SDK tool-call loop."""

import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import anthropic

from bench.models import FileUpload, RunRecord, ToolCall, ToolDef
from bench.runner.base import BaseRunner
from bench.testcase.base import TestCase

logger = logging.getLogger(__name__)


class ClaudeRunner(BaseRunner):

    def __init__(self) -> None:
        self._client = anthropic.Anthropic()
        self._messages: list[dict] = []
        self._uploaded_files: dict[str, str] = {}  # filename -> file_id
        self._text_mode = False

    def run(
        self,
        testcase: TestCase,
        model: str,
        output_dir: Path,
        followup: bool = False,
        max_turns: int = 30,
        debug: bool = False,
        text: bool = False,
    ) -> RunRecord:
        """Override run to pass testcase to _run_main and upload files."""
        self._text_mode = text
        record = RunRecord(
            provider=self._provider_name(),
            model=model,
            testcase=testcase.name,
        )

        # Update PDF metadata to bypass caching
        testcase.update_pdf_metadata()

        tools = testcase.get_tools(output_dir)

        print(f"\n=== {testcase.name} / {self._provider_name()} ===")
        print(f"Output: {output_dir.absolute()}")
        if followup and testcase.followup_prompt:
            print("Follow-up: enabled")
        print()

        t0 = time.time()
        try:
            # Phase 1: main task with testcase passed
            self._run_main(model, testcase.system_prompt, testcase.task_prompt, tools, record, max_turns, testcase)

            # Phase 2: follow-up
            if followup and testcase.followup_prompt:
                self._snapshot_outputs(output_dir)
                print(f"\n{'=' * 40}")
                print(f"  FOLLOW-UP: {testcase.followup_prompt}")
                print(f"{'=' * 40}")
                followup_text = self._run_followup(
                    model, testcase.system_prompt, testcase.followup_prompt, tools, record, max_turns,
                )
                record.followup_response = followup_text
        except Exception as e:
            record.error = str(e)
            print(f"\n[{self._provider_name()} Error] {e}")

        record.wall_time_seconds = time.time() - t0

        if output_dir.exists():
            record.output_files = [str(f) for f in output_dir.rglob("*") if f.is_file()]

        return record

    def _provider_name(self) -> str:
        return "claude"

    def _upload_and_document_files(self, data_dir: Path, input_files: list[str], testcase: TestCase | None = None, record: RunRecord | None = None) -> str:
        """Upload specified input files to Claude Files API (unless in text mode).

        Returns a formatted string documenting uploaded files and their IDs.
        """
        if not data_dir.exists() or not input_files:
            return ""

        # In text mode, don't upload files — they'll be converted to markdown
        if self._text_mode:
            print("[Claude Files API] Text mode enabled: converting files to markdown instead of uploading")
            for filename in input_files:
                file_path = data_dir / filename
                if file_path.exists():
                    # Store placeholder file_id for consistency
                    self._uploaded_files[str(file_path)] = f"text:{file_path.name}"
            return ""

        xlsx_files = []
        pdf_files = []

        # Get file checksums from testcase if available
        file_checksums = {}
        if testcase and hasattr(testcase, 'get_file_checksums'):
            file_checksums = testcase.get_file_checksums()

        for filename in input_files:
            file_path = data_dir / filename
            if not file_path.exists():
                print(f"[Claude Files API] Warning: {filename} not found in {data_dir}")
                continue

            try:
                file_id = self._upload_file(file_path, filename, file_checksums, record)
                if file_path.suffix == ".xlsx":
                    xlsx_files.append((file_path.name, file_id))
                elif file_path.suffix == ".pdf":
                    pdf_files.append((file_path.name, file_id))
            except Exception as e:
                print(f"[Claude Files API] Error uploading {file_path.name}: {e}")

        lines = []
        if xlsx_files:
            lines.append("**Excel Files:**")
            for name, fid in xlsx_files:
                lines.append(f"- {name}: `{fid}`")
        if pdf_files:
            lines.append("\n**Invoice PDFs:**")
            for name, fid in pdf_files:
                lines.append(f"- {name}: `{fid}`")

        if lines:
            return "\n".join(lines)
        return ""

    def _upload_file(self, file_path: Path, filename: str = "", file_checksums: dict | None = None, record: RunRecord | None = None) -> str:
        """Upload a single file to Claude Files API and return its file_id."""
        if str(file_path) in self._uploaded_files:
            return self._uploaded_files[str(file_path)]

        with open(file_path, "rb") as f:
            response = self._client.beta.files.upload(
                file=(file_path.name, f, self._get_mime_type(file_path)),
            )
        file_id = response.id
        self._uploaded_files[str(file_path)] = file_id

        # Log checksums if available
        if file_checksums and filename in file_checksums:
            original_checksum, updated_checksum = file_checksums[filename]
            logger.info(f"Uploaded {file_path.name} (original: {original_checksum}, updated: {updated_checksum}) → {file_id}")
            # Record file upload details in the trace
            if record:
                record.file_uploads.append(FileUpload(
                    filename=filename,
                    original_checksum=original_checksum,
                    updated_checksum=updated_checksum,
                    file_id=file_id,
                ))
        else:
            print(f"[Claude Files API] Uploaded {file_path.name} → {file_id}")

        return file_id

    def _build_message_with_files(
        self, task_prompt: str, uploaded_files: dict[str, str], data_dir: Path | None = None
    ) -> list[dict[str, Any]]:
        """Build message content with file attachments.

        If text mode is enabled: convert all files to markdown using markitdown.
        Otherwise: PDFs use document blocks with file_id, Excel files as markdown text.
        """
        import markitdown

        content: list[dict[str, Any]] = []
        file_text_parts = []

        md = markitdown.MarkItDown()

        for file_path, file_id in uploaded_files.items():
            file_path_obj = Path(file_path)
            suffix = file_path_obj.suffix.lower()
            full_path = data_dir / file_path_obj.name if data_dir else None

            if not full_path or not full_path.exists():
                file_text_parts.append(f"\n## {file_path_obj.name}\nFile not found")
                continue

            if suffix == ".pdf":
                # PDFs use document blocks with file_id
                content.append({
                    "type": "document",
                    "source": {
                        "type": "file",
                        "file_id": file_id,
                    },
                })
            elif suffix == ".xlsx":
                # Excel files: convert to markdown
                try:
                    result = md.convert(str(full_path))
                    file_text_parts.append(f"\n## {file_path_obj.name}\n{result.text_content}")
                except Exception as e:
                    file_text_parts.append(f"\n## {file_path_obj.name}\nError converting file: {e}")

        # Combine task prompt with file data if any
        combined_text = task_prompt
        if file_text_parts:
            combined_text = f"{task_prompt}\n\n### Uploaded Files (as Markdown):\n" + "".join(file_text_parts)

        # Log the full message content at debug level
        logger.debug(f"=== Message context (text portion) ===\n{combined_text}\n=== End message context ===")

        # Add the combined text
        content.append({
            "type": "text",
            "text": combined_text,
        })

        return content

    @staticmethod
    def _get_mime_type(file_path: Path) -> str:
        """Get MIME type for file."""
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            return "application/pdf"
        elif suffix == ".xlsx":
            return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        elif suffix == ".csv":
            return "text/csv"
        return "application/octet-stream"

    def _tool_loop(
        self,
        model: str,
        system: str,
        tools: list[dict[str, Any]],
        tool_fns: dict[str, Callable[..., str]],
        messages: list[dict[str, Any]],
        record: RunRecord,
        max_turns: int,
        label: str = "",
    ) -> str:
        """Run the tool-call loop, mutating messages in place. Returns assistant text."""
        tag = f"[Claude{' ' + label if label else ''}]"
        all_text: list[str] = []

        for _ in range(max_turns):
            response = self._client.beta.messages.create(
                model=model,
                max_tokens=4096,
                system=system,
                tools=tools,  # type: ignore[arg-type]
                messages=messages,  # type: ignore[arg-type]
                betas=["files-api-2025-04-14"],
            )

            # Track API request ID
            if hasattr(response, "id") and response.id:
                record.api_request_ids.append(response.id)

            record.total_tokens_in += response.usage.input_tokens
            record.total_tokens_out += response.usage.output_tokens

            for block in response.content:
                if block.type == "text" and block.text.strip():
                    print(f"\n{tag} {block.text[:500]}")
                    all_text.append(block.text)

            if response.stop_reason != "tool_use":
                messages.append({"role": "assistant", "content": response.content})
                break

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                name = block.name
                args = block.input
                record.total_tool_calls += 1
                print(f"\n  >> Tool: {name}({json.dumps(args, default=str)[:200]})")

                fn = tool_fns.get(name)
                if fn is None:
                    result = f"Error: unknown tool '{name}'"
                else:
                    try:
                        result = fn(**args)
                    except Exception as e:
                        result = f"Error: {e}"

                record.tool_calls.append(
                    ToolCall(
                        name=name,
                        arguments=args,
                        result=result[:500],
                        api_request_id=response.id,
                    )
                )
                print(f"     << {result[:200]}")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

        return "\n".join(all_text)

    def _build_tool_specs(self, tools: list[ToolDef]) -> tuple[list[dict[str, Any]], dict[str, Callable[..., str]]]:
        specs = []
        fns = {}
        for td in tools:
            specs.append({
                "name": td.name,
                "description": td.description,
                "input_schema": td.parameters,
            })
            fns[td.name] = td.fn
        return specs, fns

    def _run_main(
        self,
        model: str,
        system_prompt: str,
        task_prompt: str,
        tools: list[ToolDef],
        record: RunRecord,
        max_turns: int,
        testcase: TestCase | None = None,
    ) -> None:
        # Upload input files if testcase specifies them
        data_dir = None
        if testcase and hasattr(testcase, "data_dir") and hasattr(testcase, "input_files"):
            data_dir = testcase.get_upload_data_dir()
            self._upload_and_document_files(data_dir, testcase.input_files, testcase, record)

        specs, fns = self._build_tool_specs(tools)

        # Append run_id to task prompt to avoid prompt caching
        task_prompt_with_id = f"{task_prompt}\n\n[Run ID: {record.run_id}]"

        # Build message content with file attachments
        message_content = self._build_message_with_files(task_prompt_with_id, self._uploaded_files, data_dir)
        self._messages = [{"role": "user", "content": message_content}]
        self._tool_loop(model, system_prompt, specs, fns, self._messages, record, max_turns)

    def _run_followup(
        self,
        model: str,
        system_prompt: str,
        followup_prompt: str,
        tools: list[ToolDef],
        record: RunRecord,
        max_turns: int,
    ) -> str:
        specs, fns = self._build_tool_specs(tools)
        self._messages.append({"role": "user", "content": followup_prompt})
        return self._tool_loop(
            model, system_prompt, specs, fns, self._messages, record, max_turns, label="followup",
        )
