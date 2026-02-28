"""TestCase abstract base class."""

import hashlib
import logging
import tempfile
import time
import zipfile
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import fitz

from bench.config import DEFAULT_FOLLOWUP_PROMPT
from bench.models import ToolDef

logger = logging.getLogger(__name__)


class TestCase(ABC):
    """A benchmark test case: prompt + tools + verification.

    Subclasses must implement:
      - name (property)
      - system_prompt (property)
      - task_prompt (property)
      - get_tools(output_dir) -> list of ToolDef
      - verify(output_dir) -> dict with {structural, numerical, passed, score}

    Optional override:
      - followup_prompt (property) — defaults to generic reverification prompt
      - data_dir (property) — defaults to sibling data/ directory
    """

    def __init__(self) -> None:
        """Initialize testcase with temporary directory for updated PDFs."""
        self._temp_dir: Path | None = None
        self._file_checksums: dict[str, tuple[str, str]] = {}  # filename -> (original, updated)
        self._version: str = "v2"

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this test case (e.g. 'nrr', 'reconciliation')."""
        ...

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """System prompt for the LLM."""
        ...

    @property
    @abstractmethod
    def task_prompt(self) -> str:
        """User task prompt."""
        ...

    @property
    def followup_prompt(self) -> str:
        """Follow-up reverification prompt."""
        return DEFAULT_FOLLOWUP_PROMPT

    @property
    def data_dir(self) -> Path:
        """Path to data directory. Defaults to data/ next to the subclass module."""
        import inspect
        cls_file = inspect.getfile(type(self))
        return Path(cls_file).resolve().parent / "data"

    def set_version(self, version: str) -> None:
        """Set the data version to use (v1 or v2)."""
        self._version = version

    @property
    def input_files(self) -> list[str]:
        """List of input filenames to upload. Defaults to empty (no files)."""
        return []

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        return sha256.hexdigest()[:8]  # First 8 chars for readability

    def update_pdf_metadata(self) -> None:
        """Update PDF metadata to bypass caching. Creates temp files with updated PDFs."""
        # Only update if there are PDF files to process
        pdf_files = [f for f in self.input_files if f.lower().endswith(".pdf")]
        if not pdf_files:
            return

        # Create temporary directory
        self._temp_dir = Path(tempfile.mkdtemp())
        logger.debug(f"Created temp directory for PDFs: {self._temp_dir}")

        for filename in pdf_files:
            source_path = self.data_dir / filename
            if not source_path.exists():
                logger.warning(f"File not found: {filename}")
                continue

            try:
                logger.debug(f"Updating PDF metadata for {filename}")
                original_checksum = self._calculate_checksum(source_path)

                # Open original PDF
                doc = fitz.open(str(source_path))
                # Update metadata with timestamp to make file unique and bypass caching
                metadata = doc.metadata or {}
                metadata["producer"] = f"benchmark_{int(time.time() * 1000)}"
                doc.set_metadata(metadata)

                # Save to temporary directory
                temp_path = self._temp_dir / filename
                doc.save(str(temp_path))
                doc.close()

                updated_checksum = self._calculate_checksum(temp_path)
                self._file_checksums[filename] = (original_checksum, updated_checksum)
                logger.info(f"PDF {filename}: checksum {original_checksum} → {updated_checksum}")
                logger.debug(f"Saved updated PDF to {temp_path}")
            except Exception as e:
                logger.warning(f"Failed to update PDF metadata for {filename}: {e}")

    def update_xlsx_metadata(self) -> None:
        """Update XLSX metadata to bypass caching. Creates temp files with updated XLSXs."""
        # Only update if there are XLSX files to process
        xlsx_files = [f for f in self.input_files if f.lower().endswith(".xlsx")]
        if not xlsx_files:
            return

        # Create temporary directory if not already created
        if not self._temp_dir:
            self._temp_dir = Path(tempfile.mkdtemp())
            logger.debug(f"Created temp directory for XLSXs: {self._temp_dir}")
        else:
            logger.debug(f"Using existing temp directory: {self._temp_dir}")

        # Get current UTC time in ISO format
        now = datetime.utcnow().isoformat() + "Z"

        for filename in xlsx_files:
            source_path = self.data_dir / filename
            if not source_path.exists():
                logger.warning(f"File not found: {filename} in {self.data_dir}")
                continue

            try:
                logger.debug(f"Updating XLSX metadata for {filename}")
                original_checksum = self._calculate_checksum(source_path)

                import re
                import os

                # Extract, update, and re-zip the XLSX file
                with tempfile.TemporaryDirectory() as extract_dir:
                    # Extract XLSX (which is a ZIP)
                    with zipfile.ZipFile(str(source_path), 'r') as zip_ref:
                        zip_ref.extractall(extract_dir)

                    # Update core.xml if it exists
                    core_xml_path = Path(extract_dir) / 'docProps' / 'core.xml'
                    if core_xml_path.exists():
                        with open(core_xml_path, 'r') as f:
                            content = f.read()

                        # Update timestamps
                        content = re.sub(
                            r'<dcterms:created[^>]*>.*?</dcterms:created>',
                            f'<dcterms:created xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:type="dcterms:W3CDTF">{now}</dcterms:created>',
                            content
                        )
                        content = re.sub(
                            r'<dcterms:modified[^>]*>.*?</dcterms:modified>',
                            f'<dcterms:modified xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>',
                            content
                        )

                        with open(core_xml_path, 'w') as f:
                            f.write(content)

                    # Re-create XLSX with updated metadata
                    temp_path = self._temp_dir / filename
                    with zipfile.ZipFile(str(temp_path), 'w', zipfile.ZIP_DEFLATED) as zipf:
                        for root, dirs, files in os.walk(extract_dir):
                            for file in files:
                                file_path = Path(root) / file
                                arcname = file_path.relative_to(extract_dir)
                                zipf.write(file_path, arcname)

                updated_checksum = self._calculate_checksum(temp_path)
                self._file_checksums[filename] = (original_checksum, updated_checksum)
                logger.info(f"XLSX {filename}: checksum {original_checksum} → {updated_checksum}")
                logger.debug(f"Saved updated XLSX to {temp_path}")
            except Exception as e:
                logger.warning(f"Failed to update XLSX metadata for {filename}: {e}")

    def get_upload_data_dir(self) -> Path:
        """Return the directory containing files to upload (temp dir if PDFs were updated, else original)."""
        if self._temp_dir:
            return self._temp_dir
        return self.data_dir

    def get_file_checksums(self) -> dict[str, tuple[str, str]]:
        """Return dict of filename -> (original_checksum, updated_checksum)."""
        return self._file_checksums

    @abstractmethod
    def get_tools(self, output_dir: Path) -> list[ToolDef]:
        """Return tools with output_dir captured in closures. No os.chdir needed."""
        ...

    @abstractmethod
    def verify(self, output_dir: Path) -> dict:
        """Verify outputs against golden baselines.

        Returns dict with keys: structural, numerical, passed, score.
        """
        ...
