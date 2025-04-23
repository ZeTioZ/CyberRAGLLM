from langchain_ollama import ChatOllama

class LlmModel:
	def __init__(self, model_name: str, model_format: str):
		self.model_name: str = model_name
		self.model_format: str = model_format
		self.model: ChatOllama = ChatOllama(model=model_name, temperature=0)
		self.model_modded: ChatOllama = ChatOllama(model=model_name, temperature=0, format=model_format)

	def get_model_name(self):
		return self.model_name

	def get_model_format(self):
		return self.model_format

	def get_model(self):
		return self.model

	def get_model_modded(self):
		return self.model_modded

	def __repr__(self):
		return f"LlmModel(model_name={self.model_name}, model_format={self.model_format})"