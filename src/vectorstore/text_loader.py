"""
Module for loading and processing text-based files for the CyberRAGLLM application.

This module provides a custom text file loader that can handle various text-based
files such as Markdown (.md), plain text (.txt), and other text formats. It includes
functionality for reading local text files and downloading text content from URLs.

The module preserves the original formatting of the text files, which can be
important for structured text formats like Markdown.
"""

import os
import tempfile
from typing import List, Dict, Optional
import requests
from langchain_core.documents import Document

class TextLoader:
	"""Custom text file loader that can handle various text-based file formats.

	This class provides functionality to download text files from URLs, read content
	from local text files, and convert the content into Document objects for further
	processing. It supports various text formats including Markdown (.md), plain text (.txt),
	and other text-based formats.
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

	def _download_text(self, url: str) -> str:
		"""Download text content from URL to a temporary file."""
		response = requests.get(url, headers=self.headers, verify=self.verify)
		response.raise_for_status()

		# Create a temporary file to store the text content
		file_extension = self._get_file_extension(url)
		temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
		temp_file.write(response.content)
		temp_file.close()

		return temp_file.name

	def _get_file_extension(self, path: str) -> str:
		"""Get the file extension from the path."""
		# Extract the file extension from the path
		_, ext = os.path.splitext(path)
		if not ext:
			# Default to .txt if no extension is found
			return ".txt"
		return ext

	def _read_text_file(self, file_path: str) -> str:
		"""Read content from a text file."""
		with open(file_path, 'r', encoding='utf-8') as file:
			return file.read()

	def _get_content_format(self, file_path: str) -> str:
		"""Determine the content format based on file extension."""
		ext = self._get_file_extension(file_path).lower()
		if ext == '.md':
			return "markdown"
		elif ext == '.txt':
			return "text"
		else:
			# Default to text for other extensions
			return "text"

	def load(self) -> List[Document]:
		"""Load text data into document objects.

		This method handles both local and remote text files, reads their content,
		and returns a list of Document objects with the content and appropriate metadata.
		"""
		docs = []
		try:
			# Handle URL or local file
			if self._is_url(self.file_path):
				# Download text from URL
				temp_file_path = self._download_text(self.file_path)
				content = self._read_text_file(temp_file_path)

				# Clean up temporary file
				os.unlink(temp_file_path)

				source = self.file_path
			else:
				# Local file
				content = self._read_text_file(self.file_path)
				source = os.path.abspath(self.file_path)

			content_format = self._get_content_format(self.file_path)
			file_type = self._get_file_extension(self.file_path)[1:]  # Remove the dot

			metadata = {
				"source": source,
				"title": os.path.basename(self.file_path),
				"file_type": file_type,
				"content_format": content_format
			}

			docs.append(Document(page_content=content, metadata=metadata))
		except Exception as e:
			if self.continue_on_failure:
				print(f"Error processing text file {self.file_path}: {e}")
			else:
				raise e

		return docs