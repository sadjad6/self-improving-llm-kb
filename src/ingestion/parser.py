"""Markdown document parser with Obsidian-style feature extraction."""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

from src.utils.models import Document

logger = logging.getLogger(__name__)

# Regex patterns for Obsidian features
WIKI_LINK_PATTERN = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")
TAG_PATTERN = re.compile(r"(?:^|\s)#([a-zA-Z][a-zA-Z0-9_-]*)")
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


class MarkdownParser:
    """Parses markdown files and extracts structured information.

    Handles Obsidian-style features including wiki links ([[...]]),
    tags (#tag), headings, and YAML frontmatter.
    """

    def __init__(
        self,
        extract_links: bool = True,
        extract_tags: bool = True,
    ) -> None:
        self.extract_links = extract_links
        self.extract_tags = extract_tags

    def parse_file(self, filepath: Path) -> Document:
        """Parse a single markdown file into a Document.

        Args:
            filepath: Path to the markdown file.

        Returns:
            Parsed Document with extracted metadata.
        """
        text = filepath.read_text(encoding="utf-8")
        doc_id = hashlib.md5(str(filepath).encode()).hexdigest()[:12]
        title = self._extract_title(text, filepath)

        # Strip frontmatter for content
        content = FRONTMATTER_PATTERN.sub("", text).strip()

        headings = [m.group(2).strip() for m in HEADING_PATTERN.finditer(text)]
        links = self._extract_links(text) if self.extract_links else []
        tags = self._extract_tags(text) if self.extract_tags else []

        logger.debug(
            "Parsed %s: %d headings, %d links, %d tags",
            filepath.name, len(headings), len(links), len(tags),
        )

        return Document(
            id=doc_id,
            title=title,
            content=content,
            source_path=str(filepath),
            headings=headings,
            links=links,
            tags=tags,
        )

    def parse_directory(self, directory: Path) -> list[Document]:
        """Parse all markdown files in a directory.

        Args:
            directory: Path to directory containing .md files.

        Returns:
            List of parsed Documents.
        """
        directory = Path(directory)
        if not directory.exists():
            logger.error("Directory %s does not exist.", directory)
            return []

        md_files = sorted(directory.glob("**/*.md"))
        logger.info("Found %d markdown files in %s", len(md_files), directory)

        documents = []
        for fp in md_files:
            try:
                doc = self.parse_file(fp)
                documents.append(doc)
            except Exception as e:
                logger.error("Failed to parse %s: %s", fp, e)

        return documents

    @staticmethod
    def _extract_title(text: str, filepath: Path) -> str:
        """Extract title from first H1 heading or filename."""
        match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
        if match:
            return match.group(1).strip()
        return filepath.stem.replace("_", " ").title()

    @staticmethod
    def _extract_links(text: str) -> list[str]:
        """Extract Obsidian-style wiki links."""
        return list(set(WIKI_LINK_PATTERN.findall(text)))

    @staticmethod
    def _extract_tags(text: str) -> list[str]:
        """Extract hashtag-style tags."""
        return list(set(TAG_PATTERN.findall(text)))

