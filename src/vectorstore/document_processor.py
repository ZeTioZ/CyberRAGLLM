"""
Module for processing documents from web URLs into a vector store for retrieval.

This module provides functionality to load web content, split it into chunks,
embed those chunks using a language model, and create a vector store for
efficient semantic retrieval.
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from src.vectorstore.custom_web_loader import CustomWebLoader


class DocumentProcessor:
	"""
	A class for processing web documents into a vector store for retrieval.

	This class handles loading web content from URLs, splitting the content into chunks,
	embedding those chunks using a language model, and creating a vector store for
	efficient semantic retrieval.

	Attributes:
		urls (list): List of URLs to load content from.
		chunk_size (int): Size of text chunks for splitting documents.
		chunk_overlap (int): Overlap between text chunks.
		model (str): Name of the embedding model to use.
		inference_mode (str): Mode for inference, either "local" or "remote".
		k (int): Number of documents to retrieve.
		retriever (VectorStoreRetriever): The retriever for getting documents.
	"""
	def __init__(self, urls: list=None, chunk_size: int=1000, chunk_overlap: int=200, model: str="sentence-transformers/all-mpnet-base-v2", inference_mode: str="local", k: int=3):
		"""
		Initialize the DocumentProcessor.

		Args:
			urls (list, optional): List of URLs to load content from. Defaults to None.
			chunk_size (int, optional): Size of text chunks for splitting documents. Defaults to 1000.
			chunk_overlap (int, optional): Overlap between text chunks. Defaults to 200.
			model (str, optional): Name of the embedding model to use. Defaults to "sentence-transformers/all-mpnet-base-v2".
			inference_mode (str, optional): Mode for inference, either "local" or "remote". Defaults to "local".
			k (int, optional): Number of documents to retrieve. Defaults to 3.
		"""
		self.urls = urls if urls is not None else []
		self.chunk_size = chunk_size
		self.chunk_overlap = chunk_overlap
		self.model = model
		self.inference_mode = inference_mode
		self.k = k
		self.retriever = self._process()

	def _process(self) -> VectorStoreRetriever:
		"""
		Process the URLs into a vector store retriever.

		This method loads documents from the URLs, splits them into chunks,
		embeds the chunks using the specified model, and creates a vector store
		for efficient retrieval.

		Returns:
			VectorStoreRetriever: A retriever for getting documents from the vector store.
		"""
		# Load documents using CustomWebLoader for better content extraction
		docs = [CustomWebLoader(url).load() for url in self.urls]
		docs_list = [item for sublist in docs for item in sublist]

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
