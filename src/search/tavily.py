"""
Module for web search functionality using the Tavily API.

This module provides a wrapper around the Tavily search API to enable
web search capabilities in the CyberRAGLLM application. It handles
setting up the necessary environment variables and initializing the
search tool.
"""

import os
import getpass
from langchain_community.tools.tavily_search import TavilySearchResults


def _set_env(var: str):
	"""
	Set an environment variable if it's not already set.

	This function checks if an environment variable is set, and if not,
	prompts the user to enter a value for it.

	Args:
		var (str): The name of the environment variable to set.
	"""
	if not os.environ.get(var):
		os.environ[var] = getpass.getpass(f"{var}: ")


class TavilySearch:
	"""
	A class for web search functionality using the Tavily API.

	This class initializes the necessary environment variables and
	creates a web search tool using the Tavily API.

	Attributes:
		web_search_tool (TavilySearchResults): The web search tool.
	"""
	def __init__(self):
		"""
		Initialize the TavilySearch.

		This method sets up the necessary environment variables and
		creates a web search tool using the Tavily API and logs the process using Langsmith API.
		"""
		_set_env("TAVILY_API_KEY")
		os.environ["TOKENIZERS_PARALLELISM"] = "true"
		_set_env("LANGSMITH_API_KEY")
		os.environ["LANGCHAIN_TRACING_V2"] = "true"
		os.environ["LANGCHAIN_PROJECT"] = "CyberRAGLLM"
		self.web_search_tool = TavilySearchResults(k=3)
