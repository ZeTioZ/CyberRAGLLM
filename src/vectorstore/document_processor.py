from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings


class DocumentProcessor:
	def __init__(self, urls: list=None, chunk_size: int=1000, chunk_overlap: int=200, model: str="sentence-transformers/all-mpnet-base-v2", inference_mode: str="local", k: int=3):
		self.urls = urls if urls is not None else []
		self.chunk_size = chunk_size
		self.chunk_overlap = chunk_overlap
		self.model = model
		self.inference_mode = inference_mode
		self.k = k
		self.retriever = self._process()

	def _process(self) -> VectorStoreRetriever:
		# Load documents
		docs = [WebBaseLoader(url).load() for url in self.urls]
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
		"""Get the retriever."""
		return self.retriever


if __name__ == "__main__":
	urls_to_check = [
		"https://lilianweng.github.io/posts/2023-06-23-agent/",
		"https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
		"https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
	]
	processor = DocumentProcessor(urls_to_check)
	retriever = processor.get_retriever()