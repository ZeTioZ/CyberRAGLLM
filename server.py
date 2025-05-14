"""
Server module for the CyberRAGLLM application.

This module implements a FastAPI server that exposes OpenAI-compatible API endpoints
for the CyberRAGLLM application. It adapts the existing question-answering logic
to work with API requests and returns responses in the format expected by OpenAI API clients.
"""

import os
import sys
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.graph.control_flow import ControlFlowState
from src.llm.llm_model import LlmModel
from src.search.tavily import TavilySearch
from src.vectorstore.document_processor import DocumentProcessor

# Define Pydantic models for API request and response
class Message(BaseModel):
	role: str
	content: str

class ChatCompletionRequest(BaseModel):
	model: str
	messages: List[Message]
	temperature: Optional[float] = 0
	max_tokens: Optional[int] = None
	stream: Optional[bool] = False
	max_retries: Optional[int] = 3
	web_search_enabled: Optional[bool] = True

class ChatCompletionChoice(BaseModel):
	index: int
	message: Message
	finish_reason: str = "stop"

class ChatCompletionUsage(BaseModel):
	prompt_tokens: int
	completion_tokens: int
	total_tokens: int

class ChatCompletionResponse(BaseModel):
	id: str
	object: str = "chat.completion"
	created: int
	model: str
	choices: List[ChatCompletionChoice]
	usage: ChatCompletionUsage

# Initialize shared components
documents_urls = open("rag_urls.txt").read().splitlines()
print("Loading documents for the vector store,", len(documents_urls), "documents to load...")
document_processor = DocumentProcessor(urls=documents_urls, model="intfloat/multilingual-e5-large-instruct")
retriever = document_processor.get_retriever()
print("Initializing LLM model and web search tool...")
llm_model = LlmModel("hf.co/safe049/mistral-v0.3-7b-cybersecurity:latest", "json")
web_search_tool = TavilySearch().web_search_tool

print("Initializing CyberRAGLLM FastAPI...")
app = FastAPI(title="CyberRAGLLM API", description="OpenAI-compatible API for CyberRAGLLM")

# Add CORS middleware
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

# Function to create a new control flow state with the specified web search setting
def create_control_flow(web_search_enabled=True):
	control_flow_state = ControlFlowState(llm_model, retriever, web_search_tool, web_search_enabled)
	return control_flow_state.build_graph()

@app.get("/")
async def root():
	"""Root endpoint that returns basic API information."""
	return {
		"message": "CyberRAGLLM API is running",
		"version": "1.0.0",
		"endpoints": {
			"/v1/chat/completions": "OpenAI-compatible chat completions endpoint"
		}
	}

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
	"""
	OpenAI-compatible chat completions endpoint.

	This endpoint processes chat completion requests in the OpenAI API format,
	runs them through the CyberRAGLLM graph-based workflow, and returns responses
	in the format expected by OpenAI API clients.
	"""
	try:
		# Extract the question from the last user message
		user_messages = [msg for msg in request.messages if msg.role == "user"]
		if not user_messages:
			raise HTTPException(status_code=400, detail="No user message found in the request")

		question = user_messages[-1].content

		# Initialize state
		state = {
			"question": question,
			"documents": [],
			"web_search": "No",
			"max_retries": request.max_retries,
			"loop_step": 0,
			"generation": "",
			"answers": 0
		}

		# Create a new graph with the specified web search setting
		graph = create_control_flow(request.web_search_enabled)

		# Run the graph
		result = graph.invoke(state)

		# Extract the answer
		answer = result.get('generation', 'No answer generated.')
		answer_content = answer.text()

		# Create response in OpenAI format
		response = ChatCompletionResponse(
			id=f"chatcmpl-{os.urandom(4).hex()}",
			created=int(__import__('time').time()),
			model=request.model,
			choices=[
				ChatCompletionChoice(
					index=0,
					message=Message(
						role="assistant",
						content=answer_content
					)
				)
			],
			usage=ChatCompletionUsage(
				prompt_tokens=len(question),
				completion_tokens=len(answer_content),
				total_tokens=(len(question) + len(answer_content))
			)
		)

		return response

	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

if __name__ == "__main__":
	import uvicorn
	uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
