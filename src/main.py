"""
Main module for the CyberRAGLLM application.

This module serves as the entry point for the CyberRAGLLM application, which
uses a graph-based workflow to answer cybersecurity questions using Retrieval-Augmented
Generation (RAG) and web search. It initializes the necessary components, builds
the workflow graph, and handles user interaction.
"""

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

	This function initializes the necessary components for the CyberRAGLLM application,
	including the language model, web search tool, document processor, and control flow
	state. It then builds the workflow graph and enters a loop to process user questions.

	The function handles user input, processes the question through the graph-based
	workflow, and displays the results to the user. It continues processing questions
	until the user terminates the application.
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
		max_retries_input = input("Enter the maximum number of retries (default is 3): ")
		if max_retries_input.strip() == "":
			max_retries_input = 3
		else:
			try:
				max_retries_input = int(max_retries_input)
			except ValueError:
				print("Invalid input. Using default value of 3.")
				max_retries_input = 3

		# Initialize state
		state = {
			"question": question,
			"documents": [],
			"web_search": "No",
			"max_retries": max_retries_input,
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

if __name__ == "__main__":
	main()
