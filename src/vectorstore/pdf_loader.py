"""
Module for loading and processing PDF content for the CyberRAGLLM application.

This module provides a custom PDF loader that can handle both local and internet
PDF files. It includes functionality for downloading PDFs from URLs, reading
PDF content, and converting it to markdown format for further processing.

The module uses pymupdf4llm to extract PDF content as markdown, which preserves
more of the document's structure and formatting than plain text extraction.
This can improve the quality of the information retrieved from PDFs and make
it more useful for downstream NLP tasks.
"""

import os
import tempfile
from typing import List, Dict, Optional
import requests
import pymupdf4llm
from langchain_core.documents import Document

class PDFLoader:
	"""Custom PDF loader that can handle both local and internet PDF files.

	This class provides functionality to download PDFs from URLs, extract content
	from PDF files as markdown using pymupdf4llm, and convert the content into
	Document objects for further processing. The markdown format preserves more
	of the document's structure and formatting than plain text extraction.
	"""

	def __init__(self, file_path: str, header_template: Optional[Dict[str, str]] = None, verify_ssl: bool = True, continue_on_failure: bool = False):
		"""Initialize with file path or URL."""
		self.file_path = file_path
		self.headers = header_template.copy() if header_template else {}
		self.verify = verify_ssl
		self.continue_on_failure = continue_on_failure

		# Add a user agent if not present
		if "User-Agent" not in self.headers:
			self.headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

	def _is_url(self, path: str) -> bool:
		"""Check if the path is a URL."""
		return path.startswith("http://") or path.startswith("https://")

	def _download_pdf(self, url: str) -> str:
		"""Download PDF from URL to a temporary file."""
		response = requests.get(url, headers=self.headers, verify=self.verify)
		response.raise_for_status()

		# Create a temporary file to store the PDF
		temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
		temp_file.write(response.content)
		temp_file.close()

		return temp_file.name

	def _extract_markdown_from_pdf(self, file_path: str) -> str:
		"""Extract markdown content from a PDF file.

		Uses pymupdf4llm to convert PDF content to markdown format, which preserves
		more of the document's structure and formatting than plain text extraction.
		"""
		markdown_content = pymupdf4llm.to_markdown(file_path)
		return markdown_content

	def load(self) -> List[Document]:
		"""Load PDF data into document objects with markdown content.

		This method handles both local and remote PDF files, extracts their content
		as markdown using pymupdf4llm, and returns a list of Document objects with
		the markdown content and appropriate metadata.
		"""
		docs = []
		try:
			# Handle URL or local file
			if self._is_url(self.file_path):
				# Download PDF from URL
				temp_file_path = self._download_pdf(self.file_path)
				content = self._extract_markdown_from_pdf(temp_file_path)

				# Clean up temporary file
				os.unlink(temp_file_path)

				source = self.file_path
			else:
				# Local file
				content = self._extract_markdown_from_pdf(self.file_path)
				source = os.path.abspath(self.file_path)

			metadata = {
				"source": source,
				"title": os.path.basename(self.file_path),
				"file_type": "pdf",
				"content_format": "markdown"
			}

			docs.append(Document(page_content=content, metadata=metadata))
		except Exception as e:
			if self.continue_on_failure:
				print(f"Error processing PDF {self.file_path}: {e}")
			else:
				raise e

		return docs
