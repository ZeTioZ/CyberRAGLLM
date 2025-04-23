import os
import sys

from src.graph.control_flow import ControlFlowState
from src.llm.llm_model import LlmModel
from src.search.tavily import TavilySearch
from src.vectorstore.document_processor import DocumentProcessor

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
	"""
	Main function to run the CyberRAGLLM application.
	"""
	print("Welcome to CyberRAGLLM!")
	print("This application uses a graph-based workflow to answer questions using RAG and web search.")

	llm_model = LlmModel("llama3.2:3b-instruct-fp16", "json")
	web_search_tool = TavilySearch().web_search_tool
	document_processor = DocumentProcessor(urls=open("../rag_urls.txt").read().splitlines(), model="sentence-transformers/all-mpnet-base-v2")
	retriever = document_processor.get_retriever()
	control_flow_state = ControlFlowState(llm_model, retriever, web_search_tool)
	graph = control_flow_state.build_graph()

	# Get user question
	while True:
		question = input("Enter your question: ")

		# Initialize state
		state = {
			"question": question,
			"documents": [],
			"web_search": "No",
			"max_retries": 3,
			"loop_step": 0,
			"generation": "",
			"answers": 0
		}

		# Run the graph
		print("\nProcessing your question...")
		result = graph.invoke(state)

		# Print the result
		print("\n=== Result ===")
		print(f"Question: {question}")
		print(f"Answer: {result.get('generation', 'No answer generated.')}")

		# Print additional information if available
		if 'documents' in result:
			print(f"\nBased on {len(result['documents'])} documents")

	return result

if __name__ == "__main__":
	main()
