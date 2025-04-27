"""
Module defining the state structure for the graph-based workflow in CyberRAGLLM.

This module provides a TypedDict class that defines the structure of the state
that is passed between nodes in the graph-based workflow. The state contains
information about the user's question, retrieved documents, and other data
needed for the workflow.
"""

import operator
from typing_extensions import TypedDict
from typing import List, Annotated


class GraphState(TypedDict):
	"""
	A TypedDict class defining the state structure for the graph-based workflow.

	This class defines the structure of the state that is passed between nodes
	in the graph-based workflow. The state contains information about the user's
	question, retrieved documents, and other data needed for the workflow.

	Attributes:
		question (str): The user's question.
		generation (str): The LLM-generated answer.
		web_search (str): Binary decision ("Yes" or "No") to run web search.
		max_retries (int): Maximum number of retries for answer generation.
		answers (int): Number of answers generated.
		loop_step (int): Current step in the loop, annotated with operator.add.
		documents (List[str]): List of retrieved documents.
	"""

	question: str
	generation: str
	web_search: str
	max_retries: int
	answers: int
	loop_step: Annotated[int, operator.add]
	documents: List[str]
