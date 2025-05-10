"""
Module for processing documents from web URLs, PDF files, and text-based files into a vector store for retrieval.

This module provides functionality to load web content, PDF files, and text-based files (like .md and .txt),
split them into chunks, embed those chunks using a language model, and create a vector store for
efficient semantic retrieval.
"""

import os
import pathlib

import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from src.vectorstore.custom_web_loader import CustomWebLoader
from src.vectorstore.pdf_loader import PDFLoader
from src.vectorstore.text_loader import TextLoader


class DocumentProcessor:
	"""
	A class for processing web documents, PDF files, and text-based files into a vector store for retrieval.

	This class handles loading web content from URLs, PDF files (both local and from URLs),
	and text-based files like Markdown (.md) and plain text (.txt) files,
	splitting the content into chunks, embedding those chunks using a language model,
	and creating a vector store for efficient semantic retrieval.

	This class implements a caching mechanism to ensure documents are only loaded once
	for the same set of URLs and parameters.

	Attributes:
		urls (list): List of URLs or file paths to load content from.
		chunk_size (int): Size of text chunks for splitting documents.
		chunk_overlap (int): Overlap between text chunks.
		model (str): Name of the embedding model to use.
		inference_mode (str): Mode for inference, either "local" or "remote".
		k (int): Number of documents to retrieve.
		retriever (VectorStoreRetriever): The retriever for getting documents.
	"""
	# Class-level cache to store retrievers for specific configurations
	_cache = {}
	def __init__(self, urls: list=None, chunk_size: int=1000, chunk_overlap: int=200, model: str="intfloat/multilingual-e5-large-instruct", inference_mode: str="local", k: int=3):
		"""
		Initialize the DocumentProcessor.

		Args:
			urls (list, optional): List of URLs or file paths to load content from.
			chunk_size (int, optional): Size of text chunks for splitting documents.
			chunk_overlap (int, optional): Overlap between text chunks.
			model (str, optional): Name of the embedding model to use.
			inference_mode (str, optional): Mode for inference, either "local" or "remote".
			k (int, optional): Number of documents to retrieve.
		"""
		self.urls = urls if urls is not None else []
		self.chunk_size = chunk_size
		self.chunk_overlap = chunk_overlap
		self.model = model
		self.inference_mode = inference_mode
		self.k = k

		cache_key = self._generate_cache_key()
		if cache_key in DocumentProcessor._cache:
			print("Using cached document retriever - skipping document loading")
			self.retriever = DocumentProcessor._cache[cache_key]
		else:
			self.retriever = self._process()
			DocumentProcessor._cache[cache_key] = self.retriever

	def _generate_cache_key(self):
		"""
		Generate a unique cache key based on the current configuration.

		Returns:
			str: A hash string that uniquely identifies the current configuration.
		"""
		sorted_urls = sorted(self.urls) if self.urls else []
		config = (
			tuple(sorted_urls),
			self.chunk_size,
			self.chunk_overlap,
			self.model,
			self.inference_mode,
			self.k
		)
		return str(hash(config))

	def _is_pdf(self, path: str) -> bool:
		"""
		Check if a path or URL points to a PDF file.

		Args:
			path (str): The path or URL to check.

		Returns:
			bool: True if the path points to a PDF file, False otherwise.
		"""
		# Check if the path ends with .pdf (case insensitive)
		if path.lower().strip('"').endswith('.pdf'):
			return True
		if path.startswith('http://') or path.startswith('https://'):
			if '?type=pdf' in path.lower() or '&type=pdf' in path.lower():
				return True
			try:
				headers = requests.head(path, allow_redirects=True).headers
				content_type = headers.get('Content-Type', '').lower()
				content_disp = headers.get('Content-Disposition', '').lower()
				if 'application/pdf' in content_type:
					return True
				if '.pdf' in content_disp and 'filename' in content_disp:
					return True
			except:
				pass
		return False

	def _is_text_file(self, path: str) -> bool:
		"""
		Check if a path or URL points to a text-based file.

		Args:
			path (str): The path or URL to check.

		Returns:
			bool: True if the path points to a text-based file, False otherwise.
		"""
		# List of common text file extensions
		text_extensions = ['.txt', '.md', '.markdown', '.rst', '.csv', '.json', '.xml', '.log', '.cfg', '.ini', '.properties']

		# Check if the path ends with a text file extension (case insensitive)
		path_lower = path.lower().strip('"')
		for ext in text_extensions:
			if path_lower.endswith(ext):
				return True
		return False

	def _process(self) -> VectorStoreRetriever | None:
		"""
		Process the URLs and file paths into a vector store retriever.

		This method loads documents from the URLs and file paths, splits them into chunks,
		embeds the chunks using the specified model, and creates a vector store
		for efficient retrieval. It automatically detects PDF files, text-based files,
		and web content, and uses the appropriate loader for each.

		Returns:
			VectorStoreRetriever: A retriever for getting documents from the vector store.
		"""
		# Load documents using appropriate loaders based on the file type
		docs = []
		for index, url in enumerate(self.urls):
			url = url.strip()
			if pathlib.Path(url).is_dir():
				for file in pathlib.Path(url).glob("*"):
					file_str = file.__str__()
					if self._is_pdf(file_str):
						print(f"Loading PDF: {file_str} ({index+1}/{len(self.urls)})")
						docs.append(PDFLoader(file_str).load())
					elif self._is_text_file(file_str):
						print(f"Loading text file: {file_str} ({index+1}/{len(self.urls)})")
						docs.append(TextLoader(file_str).load())
					else:
						print(f"Skipping unsupported file type: {file_str}")
			if self._is_pdf(url):
				# Use PDFLoader for PDF files
				print(f"Loading PDF: {url} ({index+1}/{len(self.urls)})")
				docs.append(PDFLoader(url).load())
			elif self._is_text_file(url):
				# Use TextLoader for text-based files
				print(f"Loading text file: {url} ({index+1}/{len(self.urls)})")
				docs.append(TextLoader(url).load())
			else:
				# Use CustomWebLoader for web content
				print(f"Loading web content: {url} ({index+1}/{len(self.urls)})")
				docs.append(CustomWebLoader(url).load())

		# Flatten the list of documents
		docs_list = [item for sublist in docs for item in sublist]

		if not docs_list:
			print("Warning: No documents were loaded.")
			# Return an empty retriever or handle this case as appropriate
			return None

		# Split documents
		text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
			chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
		)
		doc_splits = text_splitter.split_documents(docs_list)

		# Add to vectorDB
		vectorstore = SKLearnVectorStore.from_documents(
			documents=doc_splits,
			embedding=HuggingFaceEmbeddings(model_name=self.model)
		)

		# Create retriever
		self.retriever = vectorstore.as_retriever(k=self.k)
		return self.retriever

	def get_retriever(self) -> VectorStoreRetriever:
		"""
		Get the vector store retriever.

		Returns:
			VectorStoreRetriever: The retriever for getting documents from the vector store.
			This retriever can be used to find documents relevant to a query.
		"""
		return self.retriever


if __name__ == "__main__":
	urls_to_check = [
		"https://memoire.donatog.eu/en/bible",
		"https://memoire.donatog.eu/en/Bypassing_Data_Execution_Prevention_on_Microsoft_Windows_XP_SP2",
		"https://memoire.donatog.eu/en/hacking-blind",
		"https://memoire.donatog.eu/en/Go_Go_Gadget_Hammer-Flipping_Nested_Pointers_for_Arbitrary_Data_Leakage",
		"https://memoire.donatog.eu/en/advanced-automated-SQL-injection-attacks-and-defensive-mechanisms"
	]
	processor = DocumentProcessor(urls_to_check)
	retriever = processor.get_retriever()
	print(retriever.invoke("What is a sql injection?"))
