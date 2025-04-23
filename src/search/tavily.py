import os
import getpass
from langchain_community.tools.tavily_search import TavilySearchResults


def _set_env(var: str):
	if not os.environ.get(var):
		os.environ[var] = getpass.getpass(f"{var}: ")


class TavilySearch:
	def __init__(self):
		_set_env("TAVILY_API_KEY")
		os.environ["TOKENIZERS_PARALLELISM"] = "true"
		_set_env("LANGSMITH_API_KEY")
		os.environ["LANGCHAIN_TRACING_V2"] = "true"
		os.environ["LANGCHAIN_PROJECT"] = "CyberRAGLLM"
		self.web_search_tool = TavilySearchResults(k=3)

