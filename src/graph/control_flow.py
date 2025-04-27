"""
Module for defining the control flow graph in the CyberRAGLLM application.

This module provides the ControlFlowState class that builds and manages the
graph-based workflow for processing user questions, retrieving relevant documents,
generating answers, and evaluating the quality of those answers. The workflow
includes steps for routing questions, retrieving documents, grading document
relevance, generating answers, and checking for hallucinations.
"""

import json

from langchain_community.tools import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.vectorstores import VectorStoreRetriever
from langgraph.graph import StateGraph
from langgraph.graph import END
from langchain.schema import Document
from langgraph.graph.state import CompiledStateGraph

from src.graph.graph_state import GraphState
from src.llm.llm_model import LlmModel

RAG_PROMPT = """You are a world-class cybersecurity expert and a top-tier Capture The Flag (CTF) competitor, specializing in cryptography, reverse engineering, forensics, web exploitation, binary exploitation, OSINT, and steganography. 

Your response will be compared against multiple AI models, so maximize depth, clarity, and expert-level detail. Assume the user wants a response that stands out in accuracy, completeness, and readability.

The user needs to provide a detailed solution to the challenge.

Here is the context to use to answer the question:

{context}

Now, review the user question:

{question}

Provide an answer to this questions using only the above context.

Follow these steps to structure your answer:

1. Identify the Challenge Type
- Classify the challenge into the most relevant CTF category.  
- Justify your classification with reasoning based on key hints in the challenge description.  

2. Provide the Full Solution First
- Present the full solution immediately, ensuring it is actionable and complete.
- Include any necessary code, scripts, or commands.
- Offer at least two alternative approaches if possible.
- If external tools are required, specify installation steps and usage examples.

3. Step-by-Step Breakdown
- Explain the logic and methodology behind the solution.  
- Justify why each step is necessary and how it contributes to solving the challenge.  
- Provide insights from real-world cybersecurity experience.

4. Additional Resources & Tools
- Suggest relevant tools, frameworks, and utilities that could assist.  
- Provide links to official documentation, tutorials, and cheat sheets.

Answer:"""

ROUTER_INSTRUCTIONS = """You are an expert at routing a user question to a vectorstore or web search.

The vectorstore contains documents related to cryptography, reverse engineering, forensics, web exploitation, binary exploitation, OSINT, and steganography.

Use the vectorstore for questions on these topics. For all else, and especially for current events, use web-search.

Return JSON with single key, datasource, that is 'websearch' or 'vectorstore' depending on the question."""

DOC_GRADER_INSTRUCTIONS = """You are a grader assessing relevance of a retrieved document to a user question.

If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant."""

DOC_GRADER_PROMPT = """Here is the retrieved document: \n\n {document} \n\n Here is the user question: \n\n {question}. 

This carefully and objectively assess whether the document contains at least some information that is relevant to the question.

Return JSON with single key, binary_score, that is 'yes' or 'no' score to indicate whether the document contains at least some information that is relevant to the question."""

HALLUCINATION_GRADER_INSTRUCTIONS = """You are an expert evaluator in cybersecurity and AI model assessment. Your task is to grade the response of a Large Language Model (LLM) against the FACTS given to you.

You will be given FACTS and a LLM ANSWER.

Here is the grade criteria to follow:

(1) Ensure the LLM ANSWER is grounded in the FACTS. 

(2) Ensure the LLM ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

Score:

A score of yes means that the llm's answer meets all of the criteria. This is the highest (best) score. 

A score of no means that the llm's answer does not meet all of the criteria. This is the lowest possible score you can give.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""

HALLUCINATION_GRADER_PROMPT = """FACTS: \n\n {documents} \n\n LLM ANSWER: {generation}. 

Return JSON with two two keys, binary_score is 'yes' or 'no' score to indicate whether the LLM ANSWER is grounded in the FACTS. And a key, explanation, that contains an explanation of the score."""

ANSWER_GRADER_INSTRUCTIONS ="""You are an expert evaluator in cybersecurity and AI model assessment. Your task is to grade the response of a Large Language Model (LLM).  

You will be given a QUESTION and a LLM ANSWER.

Here are the grade criteria to follow:

1. Solution Accuracy: Compare the LLM's response to the actual solution. Score based on correctness, with partial correctness considered.
2. Methodology Used: Is the approach logical, structured, and reproducible?
3. Reproducibility: Can another person follow the same approach?
4. Quality of Reasoning: Are the explanations clear, with well-justified choices?

The llm can receive a score of yes if the answer contains extra information that is not explicitly asked for in the question.

A score of no means that the llm's answer does not meet all of the criteria. This is the lowest possible score you can give.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""

# Grader prompt
ANSWER_GRADER_PROMPT = """QUESTION: \n\n {question} \n\n LLM ANSWER: {generation}.

Return JSON with two keys, binary_score is 'yes' or 'no' score to indicate whether the LLM ANSWER meets the criteria. And a key, explanation, that contains an explanation of the score."""


def format_docs(docs):
	return "\n\n".join(doc.page_content for doc in docs)


class ControlFlowState:
	"""
	A class for managing the control flow graph in the CyberRAGLLM application.

	This class builds and manages the graph-based workflow for processing user
	questions, retrieving relevant documents, generating answers, and evaluating
	the quality of those answers. The workflow includes steps for routing questions,
	retrieving documents, grading document relevance, generating answers, and
	checking for hallucinations.

	Attributes:
		llm_model (LlmModel): The language model to use for generating answers.
		retriever (VectorStoreRetriever): The retriever for getting documents.
		web_search_tool (TavilySearchResults): The web search tool.
		workflow (StateGraph): The state graph for the workflow.
	"""
	def __init__(self, llm_model: LlmModel, retriever: VectorStoreRetriever, web_search_tool: TavilySearchResults):
		self.llm_model = llm_model
		self.retriever = retriever
		self.web_search_tool = web_search_tool
		self.workflow = StateGraph(GraphState)

	def retrieve_documents(self, state):
		"""
		Retrieve documents from vectorstore

		Args:
			state (dict): The current graph state

		Returns:
			state (dict): New key added to state, documents, that contains retrieved documents
		"""
		print("---RETRIEVE---")
		question = state["question"]

		# Write retrieved documents to documents key in state
		documents = self.retriever.invoke(question)
		return {"documents": documents}

	def generate_answer(self, state):
		"""
		Generate answer using RAG on retrieved documents

		Args:
			state (dict): The current graph state

		Returns:
			state (dict): New key added to state, generation, that contains LLM generation
		"""
		print("---GENERATE---")
		question = state["question"]
		documents = state["documents"]
		loop_step = state.get("loop_step", 0)

		# RAG generation
		docs_txt = format_docs(documents)
		rag_prompt_formatted = RAG_PROMPT.format(context=docs_txt, question=question)
		generation = self.llm_model.model.invoke([HumanMessage(content=rag_prompt_formatted)])
		return {"generation": generation, "loop_step": loop_step + 1}

	def grade_documents(self, state):
		"""
		Determines whether the retrieved documents are relevant to the question
		If any document is not relevant, we will set a flag to run web search

		Args:
			state (dict): The current graph state

		Returns:
			state (dict): Filtered out irrelevant documents and updated web_search state
		"""

		print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
		question = state["question"]
		documents = state["documents"]

		# Score each doc
		filtered_docs = []
		web_search = "No"
		for d in documents:
			doc_grader_prompt_formatted = DOC_GRADER_PROMPT.format(
				document=d.page_content, question=question
			)
			result = self.llm_model.model_formatted.invoke(
				[SystemMessage(content=DOC_GRADER_INSTRUCTIONS)]
				+ [HumanMessage(content=doc_grader_prompt_formatted)]
			)
			grade = json.loads(result.content)["binary_score"]
			# Document relevant
			if grade.lower() == "yes":
				print("---GRADE: DOCUMENT RELEVANT---")
				filtered_docs.append(d)
			# Document not relevant
			else:
				print("---GRADE: DOCUMENT NOT RELEVANT---")
				# We do not include the document in filtered_docs
				# We set a flag to indicate that we want to run web search
				web_search = "Yes"
				continue
		return {"documents": filtered_docs, "web_search": web_search}

	def web_search(self, state):
		"""
		Web search based based on the question

		Args:
			state (dict): The current graph state

		Returns:
			state (dict): Appended web results to documents
		"""

		print("---WEB SEARCH---")
		question = state["question"]
		documents = state.get("documents", [])

		# Web search
		docs = self.web_search_tool.invoke({"query": question})
		web_results = "\n".join([d["content"] for d in docs if "content" in d])
		web_results = Document(page_content=web_results)
		documents.append(web_results)
		return {"documents": documents}

	def route_question(self, state):
		"""
		Route question to web search or RAG

		Args:
			state (dict): The current graph state

		Returns:
			str: Next node to call
		"""

		print("---ROUTE QUESTION---")
		route_question = self.llm_model.model_formatted.invoke(
			[SystemMessage(content=ROUTER_INSTRUCTIONS)]
			+ [HumanMessage(content=state["question"])]
		)
		print(route_question.content)
		json_content = json.loads(route_question.content)
		if "datasource" in json_content:
			source = json_content["datasource"]
			if source == "websearch":
				print("---ROUTE QUESTION TO WEB SEARCH---")
				return "websearch"
			elif source == "vectorstore":
				print("---ROUTE QUESTION TO RAG---")
				return "vectorstore"
		return "websearch"

	def decide_to_generate(self, state):
		"""
		Determines whether to generate an answer, or add web search

		Args:
			state (dict): The current graph state

		Returns:
			str: Binary decision for next node to call
		"""

		print("---ASSESS GRADED DOCUMENTS---")
		web_search = state["web_search"]

		if web_search == "Yes":
			# All documents have been filtered check_relevance
			# We will re-generate a new query
			print(
				"---DECISION: NOT ALL DOCUMENTS ARE RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
			)
			return "websearch"
		else:
			# We have relevant documents, so generate answer
			print("---DECISION: GENERATE---")
			return "generate"


	def grade_generation_v_documents_and_question(self, state):
		"""
		Determines whether the generation is grounded in the document and answers question

		Args:
			state (dict): The current graph state

		Returns:
			str: Decision for next node to call
		"""

		print("---CHECK HALLUCINATIONS---")
		question = state["question"]
		documents = state["documents"]
		generation = state["generation"]
		max_retries = state.get("max_retries", 3)  # Default to 3 if not provided

		hallucination_grader_prompt_formatted = HALLUCINATION_GRADER_PROMPT.format(
			documents=format_docs(documents), generation=generation.content
		)
		result = self.llm_model.model_formatted.invoke(
			[SystemMessage(content=HALLUCINATION_GRADER_INSTRUCTIONS)]
			+ [HumanMessage(content=hallucination_grader_prompt_formatted)]
		)
		grade = json.loads(result.content)["binary_score"]

		# Check hallucination
		if grade == "yes":
			print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
			# Check question-answering
			print("---GRADE GENERATION vs QUESTION---")
			# Test using question and generation from above
			answer_grader_prompt_formatted = ANSWER_GRADER_PROMPT.format(
				question=question, generation=generation.content
			)
			result = self.llm_model.model_formatted.invoke(
				[SystemMessage(content=ANSWER_GRADER_INSTRUCTIONS)]
				+ [HumanMessage(content=answer_grader_prompt_formatted)]
			)
			grade = json.loads(result.content)["binary_score"]
			if grade == "yes":
				print("---DECISION: GENERATION ADDRESSES QUESTION---")
				return "useful"
			elif state["loop_step"] <= max_retries:
				print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
				return "not useful"
			else:
				print("---DECISION: MAX RETRIES REACHED---")
				return "max retries"
		elif state["loop_step"] <= max_retries:
			print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
			return "not supported"
		else:
			print("---DECISION: MAX RETRIES REACHED---")
			return "max retries"

	def build_graph(self) -> CompiledStateGraph:
		"""
		Build the graph for the control flow
		"""
		workflow = self.workflow
		workflow.add_node("websearch", self.web_search)  # web search
		workflow.add_node("retrieve", self.retrieve_documents)  # retrieve
		workflow.add_node("grade_documents", self.grade_documents)  # grade documents
		workflow.add_node("generate", self.generate_answer)  # generate

		# Build graph
		workflow.set_conditional_entry_point(
			self.route_question,
			{
				"websearch": "websearch",
				"vectorstore": "retrieve",
			},
		)
		workflow.add_edge("websearch", "generate")
		workflow.add_edge("retrieve", "grade_documents")
		workflow.add_conditional_edges(
			"grade_documents",
			self.decide_to_generate,
			{
				"websearch": "websearch",
				"generate": "generate",
			},
		)
		workflow.add_conditional_edges(
			"generate",
			self.grade_generation_v_documents_and_question,
			{
				"not supported": "generate",
				"useful": END,
				"not useful": "websearch",
				"max retries": END,
			},
		)
		graph = workflow.compile()
		return graph
