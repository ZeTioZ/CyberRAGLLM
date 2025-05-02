"""
Module for processing documents from web URLs and PDF files into a vector store for retrieval.

This module provides functionality to load web content and PDF files, split them into chunks,
embed those chunks using a language model, and create a vector store for
efficient semantic retrieval.
"""

import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from src.vectorstore.custom_web_loader import CustomWebLoader
from src.vectorstore.pdf_loader import PDFLoader


class DocumentProcessor:
	"""
	A class for processing web documents and PDF files into a vector store for retrieval.

	This class handles loading web content from URLs and PDF files (both local and from URLs),
	splitting the content into chunks, embedding those chunks using a language model,
	and creating a vector store for efficient semantic retrieval.

	Attributes:
		urls (list): List of URLs or file paths to load content from.
		chunk_size (int): Size of text chunks for splitting documents.
		chunk_overlap (int): Overlap between text chunks.
		model (str): Name of the embedding model to use.
		inference_mode (str): Mode for inference, either "local" or "remote".
		k (int): Number of documents to retrieve.
		retriever (VectorStoreRetriever): The retriever for getting documents.
	"""
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
		self.retriever = self._process()

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

	def _process(self) -> VectorStoreRetriever:
		"""
		Process the URLs and file paths into a vector store retriever.

		This method loads documents from the URLs and file paths, splits them into chunks,
		embeds the chunks using the specified model, and creates a vector store
		for efficient retrieval. It automatically detects PDF files and uses the
		appropriate loader.

		Returns:
			VectorStoreRetriever: A retriever for getting documents from the vector store.
		"""
		# Load documents using appropriate loaders based on the file type
		docs = []
		for index, url in enumerate(self.urls):
			if self._is_pdf(url):
				# Use PDFLoader for PDF files
				print(f"Loading PDF: {url}")
				docs.append(PDFLoader(url).load())
			else:
				# Use CustomWebLoader for web content
				print(f"Loading web content: {url}")
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
