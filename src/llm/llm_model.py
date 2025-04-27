"""
Module for managing language models in the CyberRAGLLM application.

This module provides a wrapper around the ChatOllama model to simplify
initialization and access to different configurations of the model.
"""

from langchain_ollama import ChatOllama

class LlmModel:
	"""
	A class for managing language models in the CyberRAGLLM application.

	This class provides a wrapper around the ChatOllama model, allowing for
	easy initialization and access to different configurations of the model.

	Attributes:
		model_name (str): The name of the language model to use.
		model_format (str): The format for model output (e.g., "json").
		model (ChatOllama): The standard ChatOllama model instance.
		model_formatted (ChatOllama): The ChatOllama model instance with formatting options.
	"""
	def __init__(self, model_name: str, model_format: str):
		"""
		Initialize the LlmModel.

		Args:
			model_name (str): The name of the language model to use.
			model_format (str): The format for model output (e.g., "json").
		"""
		self.model_name: str = model_name
		self.model_format: str = model_format
		self.model: ChatOllama = ChatOllama(model=model_name, temperature=0)
		self.model_formatted: ChatOllama = ChatOllama(model=model_name, temperature=0, format=model_format)

	def get_model_name(self) -> str:
		"""
		Get the name of the language model.

		Returns:
			str: The name of the language model.
		"""
		return self.model_name

	def get_model_format(self) -> str:
		"""
		Get the format for model output.

		Returns:
			str: The format for model output (e.g., "json").
		"""
		return self.model_format

	def get_model(self) -> ChatOllama:
		"""
		Get the standard ChatOllama model instance.

		Returns:
			ChatOllama: The standard ChatOllama model instance.
		"""
		return self.model

	def get_model_modded(self) -> ChatOllama:
		"""
		Get the ChatOllama model instance with formatting options.

		Returns:
			ChatOllama: The ChatOllama model instance with formatting options.
		"""
		return self.model_formatted

	def __repr__(self) -> str:
		"""
		Get a string representation of the LlmModel.

		Returns:
			str: A string representation of the LlmModel.
		"""
		return f"LlmModel(model_name={self.model_name}, model_format={self.model_format})"
