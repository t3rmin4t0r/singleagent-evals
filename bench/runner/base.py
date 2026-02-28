"""BaseRunner ABC — timing, snapshot, output collection, error catching."""

import shutil
import time
from abc import ABC, abstractmethod
from pathlib import Path

from bench.models import RunRecord, ToolDef
from bench.testcase.base import TestCase


class BaseRunner(ABC):
    """Abstract runner. Subclasses implement _run_main and _run_followup."""

    @abstractmethod
    def _run_main(
        self,
        model: str,
        system_prompt: str,
        task_prompt: str,
        tools: list[ToolDef],
        record: RunRecord,
        max_turns: int,
    ) -> None:
        """Run the main task. Mutates record in place."""
        ...

    @abstractmethod
    def _run_followup(
        self,
        model: str,
        system_prompt: str,
        followup_prompt: str,
        tools: list[ToolDef],
        record: RunRecord,
        max_turns: int,
    ) -> str:
        """Run follow-up in the same conversation. Returns followup text."""
        ...

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
        """Run a test case end-to-end with timing and error handling."""
        record = RunRecord(
            provider=self._provider_name(),
            model=model,
            testcase=testcase.name,
        )

        tools = testcase.get_tools(output_dir)

        print(f"\n=== {testcase.name} / {self._provider_name()} ===")
        print(f"Output: {output_dir.absolute()}")
        if followup and testcase.followup_prompt:
            print("Follow-up: enabled")
        print()

        t0 = time.time()
        try:
            # Phase 1: main task
            self._run_main(model, testcase.system_prompt, testcase.task_prompt, tools, record, max_turns)

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

    @abstractmethod
    def _provider_name(self) -> str:
        ...

    @staticmethod
    def _snapshot_outputs(output_dir: Path) -> None:
        """Copy all CSVs from output_dir to a pre_followup sibling directory."""
        pre_dir = output_dir.parent / "pre_followup"
        pre_dir.mkdir(exist_ok=True)
        for csv_file in output_dir.rglob("*.csv"):
            shutil.copyfile(csv_file, pre_dir / csv_file.name)
