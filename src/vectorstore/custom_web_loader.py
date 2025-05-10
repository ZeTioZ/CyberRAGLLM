"""
Module for loading and processing web content for the CyberRAGLLM application.

This module provides a custom web loader that extends the WebBaseLoader from
langchain_community to extract content from web pages more effectively. It
includes functionality for scraping web pages, extracting meaningful content,
and handling various edge cases like content stored in JavaScript variables.
"""

from langchain_community.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
import requests
import re

class CustomWebLoader(WebBaseLoader):
	"""Custom web loader that extracts content from web pages more effectively."""

	def __init__(
			self,
			web_path: str,
			header_template: Optional[Dict[str, str]] = None,
			verify_ssl: bool = True,
			continue_on_failure: bool = False,
			**kwargs: Any,
	):
		"""Initialize with web path."""
		super().__init__(
			web_path=web_path,
			header_template=header_template,
			verify_ssl=verify_ssl,
			continue_on_failure=continue_on_failure,
			**kwargs,
		)
		# Store parameters locally
		self.headers = header_template.copy() if header_template else {}
		self.verify = verify_ssl
		self.continue_on_failure = continue_on_failure

		# Add a user agent if not present
		if "User-Agent" not in self.headers:
			self.headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

	def _scrape(self, url: str) -> str:
		"""Scrape the content from the URL."""
		response = requests.get(url, headers=self.headers, verify=self.verify)
		response.raise_for_status()

		return response.text

	def _extract_content(self, html: str) -> str:
		"""Extract content from HTML, looking for large text blocks."""
		soup = BeautifulSoup(html, "html.parser")

		# Get all text elements
		all_text_elements = soup.find_all(string=True)

		# Filter out small text blocks, script tags, style tags, etc.
		content_blocks = []
		for text in all_text_elements:
			parent = text.parent.name if text.parent else None
			if parent and parent.lower() in ['script', 'style', 'meta', 'noscript']:
				continue

			text_content = text.strip()
			if len(text_content) > 100:  # Only include substantial text blocks
				content_blocks.append(text_content)

		# If we found content blocks, join them
		if content_blocks:
			return "\n\n".join(content_blocks)

		# If no content blocks were found, try to extract from script tags
		# This is a fallback for sites that store content in JavaScript variables
		script_tags = soup.find_all('script')
		for script in script_tags:
			script_content = script.string
			if script_content:
				# Look for content in JSON-like structures
				content_matches = re.findall(r'"content"\s*:\s*"([^"]+)"', script_content)
				if content_matches:
					return "\n\n".join(content_matches)

		# If all else fails, return the title at least
		title = soup.title.string if soup.title else ""
		return title

	def load(self) -> List[Document]:
		"""Load data into document objects."""
		docs = []
		for url in self.web_paths:
			try:
				html = self._scrape(url)
				content = self._extract_content(html)

				# Extract metadata
				soup = BeautifulSoup(html, "html.parser")
				title = soup.title.string if soup.title else ""

				# Get description from meta tags
				description = ""
				desc_tag = soup.find("meta", attrs={"name": "description"})
				if desc_tag and "content" in desc_tag.attrs:
					description = desc_tag["content"]

				# Get language from html tag
				language = "en"  # Default
				html_tag = soup.find("html")
				if html_tag and "lang" in html_tag.attrs:
					language = html_tag["lang"]

				metadata = {
					"source": url,
					"title": title,
					"description": description,
					"language": language,
				}

				docs.append(Document(page_content=content, metadata=metadata))
			except Exception as e:
				if self.continue_on_failure:
					print(f"Error fetching or processing {url}: {e}")
					continue
				else:
					raise e

		return docs
