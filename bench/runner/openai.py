"""OpenAI runner — Agents SDK."""

import hashlib
import logging
import time
from pathlib import Path
from typing import Any

from agents import Agent, Runner, function_tool
from agents.items import ToolCallItem
from openai import OpenAI

from bench.models import FileUpload, RunRecord, ToolCall, ToolDef
from bench.runner.base import BaseRunner
from bench.testcase.base import TestCase

logger = logging.getLogger(__name__)


class OpenAIRunner(BaseRunner):

    def __init__(self) -> None:
        self._agent: Agent | None = None
        self._last_result = None
        self._openai_client = OpenAI()
        self._uploaded_files: dict[str, str] = {}  # filename -> file_id

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
        logger.debug(f"Starting main task with {len(tools)} tools")
        try:
            # Phase 1: main task with testcase passed
            logger.debug("Calling _run_main...")
            self._run_main(model, testcase.system_prompt, testcase.task_prompt, tools, record, max_turns, testcase)
            logger.debug("_run_main completed")

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
        return "openai"

    def _upload_and_document_files(self, data_dir: Path, input_files: list[str], testcase: TestCase | None = None, record: RunRecord | None = None) -> str:
        """Upload specified input files to OpenAI Files API.

        Returns a formatted string documenting uploaded files and their IDs.
        """
        logger.debug(f"_upload_and_document_files: data_dir={data_dir}, input_files={input_files}")
        if not data_dir.exists() or not input_files:
            logger.debug(f"Skipping file upload: data_dir_exists={data_dir.exists()}, has_files={bool(input_files)}")
            return ""

        # Get file checksums from testcase if available
        file_checksums = {}
        if testcase:
            file_checksums = testcase.get_file_checksums()

        xlsx_files = []
        pdf_files = []

        for filename in input_files:
            file_path = data_dir / filename
            if not file_path.exists():
                logger.warning(f"File not found: {filename} in {data_dir}")
                continue

            try:
                logger.debug(f"Uploading file: {filename}")
                file_id = self._upload_file(file_path, filename, file_checksums, record)
                logger.debug(f"Uploaded {filename} → {file_id}")
                if file_path.suffix == ".xlsx":
                    xlsx_files.append((file_path.name, file_id))
                elif file_path.suffix == ".pdf":
                    pdf_files.append((file_path.name, file_id))
            except Exception as e:
                logger.error(f"Error uploading {file_path.name}: {e}", exc_info=True)

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

    def _upload_file(self, file_path: Path, filename: str = "", file_checksums: dict[str, tuple[str, str]] | None = None, record: RunRecord | None = None) -> str:
        """Upload a single file to OpenAI Files API and return its file_id."""
        if str(file_path) in self._uploaded_files:
            logger.debug(f"File already uploaded: {file_path.name} → {self._uploaded_files[str(file_path)]}")
            return self._uploaded_files[str(file_path)]

        # Calculate checksum for logging
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        current_checksum = sha256.hexdigest()[:8]

        # Get original and updated checksums if available
        original_checksum = current_checksum
        updated_checksum = current_checksum
        if file_checksums and filename in file_checksums:
            original_checksum, updated_checksum = file_checksums[filename]

        logger.debug(f"Uploading to OpenAI Files API: {file_path.name}")
        with open(file_path, "rb") as f:
            response = self._openai_client.files.create(
                file=(file_path.name, f, self._get_mime_type(file_path)),
                purpose="user_data",
            )
        file_id = response.id
        self._uploaded_files[str(file_path)] = file_id
        logger.info(f"Uploaded {file_path.name} (original: {original_checksum}, updated: {updated_checksum}) → {file_id}")

        # Record file upload in trace
        if record:
            record.file_uploads.append(FileUpload(
                filename=filename or file_path.name,
                original_checksum=original_checksum,
                updated_checksum=updated_checksum,
                file_id=file_id,
            ))

        return file_id

    def _build_input_with_files(self, task_prompt: str, uploaded_files: dict[str, str]) -> list[dict[str, Any]]:
        """Build input with file attachments in the format OpenAI API expects."""
        content: list[dict[str, Any]] = []

        # Add file references first
        for file_path, file_id in uploaded_files.items():
            content.append({
                "type": "input_file",
                "file_id": file_id,
            })

        # Add the task prompt text
        content.append({
            "type": "input_text",
            "text": task_prompt,
        })

        return [{
            "role": "user",
            "content": content,
        }]

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

    def _build_agent(self, model: str, system_prompt: str, tools: list[ToolDef]) -> Agent:
        agent_tools = [
            function_tool(td.fn, name_override=td.name, description_override=td.description)
            for td in tools
        ]
        return Agent(
            name="benchmark_agent",
            instructions=system_prompt,
            model=model,
            tools=agent_tools,
        )

    def _extract_usage(self, result, record: RunRecord) -> None:
        # Extract request IDs from raw responses
        request_id_by_index: dict[int, str] = {}
        for i, resp in enumerate(result.raw_responses):
            # Capture API request ID
            if hasattr(resp, "id") and resp.id:
                record.api_request_ids.append(resp.id)
                request_id_by_index[i] = resp.id

            if hasattr(resp, "usage") and resp.usage:
                usage = resp.usage
                record.total_tokens_in += getattr(usage, "input_tokens", 0) or getattr(usage, "prompt_tokens", 0) or 0
                record.total_tokens_out += getattr(usage, "output_tokens", 0) or getattr(usage, "completion_tokens", 0) or 0

        # Extract tool calls and associate with request IDs
        # Get the last request ID for association (most recent API call)
        last_request_id = list(request_id_by_index.values())[-1] if request_id_by_index else None

        for item in result.new_items:
            if isinstance(item, ToolCallItem):
                record.total_tool_calls += 1
                name = getattr(item, "name", "") or getattr(item, "tool_name", "") or ""
                args = getattr(item, "arguments", "") or ""
                output = getattr(item, "output", "") or ""
                record.tool_calls.append(
                    ToolCall(
                        name=name,
                        arguments={"raw": args},
                        result=str(output)[:500],
                        api_request_id=last_request_id,
                    )
                )

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
        logger.debug(f"_run_main: model={model}, testcase={testcase.name if testcase else None}")

        # Upload input files if testcase specifies them
        if testcase and hasattr(testcase, 'data_dir') and hasattr(testcase, 'input_files'):
            upload_dir = testcase.get_upload_data_dir()
            logger.debug(f"Uploading files from {upload_dir}")
            self._upload_and_document_files(upload_dir, testcase.input_files, testcase, record)
            logger.debug(f"Files uploaded: {list(self._uploaded_files.keys())}")

        logger.debug(f"Building agent with model={model}")
        self._agent = self._build_agent(model, system_prompt, tools)
        logger.debug("Agent built successfully")

        # Append run_id to task prompt to avoid prompt caching
        task_prompt_with_id = f"{task_prompt}\n\n[Run ID: {record.run_id}]"

        # Build input with file attachments
        logger.debug(f"Building input with {len(self._uploaded_files)} files")
        input_with_files = self._build_input_with_files(task_prompt_with_id, self._uploaded_files)
        logger.debug(f"Input structure: {len(input_with_files)} messages, content has {len(input_with_files[0]['content'])} items")

        logger.debug("Calling Runner.run_sync...")
        self._last_result = Runner.run_sync(self._agent, input_with_files)  # type: ignore[arg-type]
        logger.debug("Runner.run_sync completed")

        print(f"\n[OpenAI] {str(self._last_result.final_output)[:500]}")
        self._extract_usage(self._last_result, record)

    def _run_followup(
        self,
        model: str,
        system_prompt: str,
        followup_prompt: str,
        tools: list[ToolDef],
        record: RunRecord,
        max_turns: int,
    ) -> str:
        logger.debug("_run_followup called")
        if self._last_result is None or self._agent is None:
            logger.error("Cannot run followup: no previous result or agent")
            return "Error: No previous result to follow up on"

        logger.debug("Building followup input")
        followup_input = self._last_result.to_input_list() + [{"role": "user", "content": followup_prompt}]
        logger.debug("Calling Runner.run_sync for followup...")
        result = Runner.run_sync(self._agent, followup_input)  # type: ignore[arg-type]
        logger.debug("Followup Runner.run_sync completed")
        followup_text = str(result.final_output)
        print(f"\n[OpenAI followup] {followup_text[:500]}")
        self._extract_usage(result, record)
        self._last_result = result
        return followup_text
