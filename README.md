# CyberRAGLLM

CyberRAGLLM is a cybersecurity-focused question-answering system that uses Retrieval-Augmented Generation (RAG) with local LLMs to provide accurate and contextually relevant responses to cybersecurity questions. It leverages a graph-based workflow to combine document retrieval, web search, and quality control mechanisms.

## Features

- 🔒 **Cybersecurity Focus**: Specialized for cybersecurity domain with extensive document collection
- 🤖 **Local LLM Integration**: Works with Ollama-based local LLMs (e.g., Llama 3.2)
- 📚 **Web Document Processing**: Loads and processes documents from web URLs
- 🔍 **Vector Search**: Efficient retrieval using SKLearnVectorStore with HuggingFace embeddings
- 🌐 **Web Search Integration**: Uses Tavily Search API for supplementary information
- 🧠 **Graph-based Workflow**: Sophisticated control flow for question routing and answer generation
- 🧐 **Quality Control**: Grades document relevance and answer quality
- 📊 **Hallucination Detection**: Checks if generated answers are grounded in the retrieved documents
- 🔄 **Adaptive Retrieval**: Combines vector search with web search when needed

## Installation

1. Clone the repository:
```bash
git clone https://github.com/zetioz/CyberRAGLLM.git
cd CyberRAGLLM
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv .venv
.\.venv\Scripts\activate  # On Windows
# OR
source .venv/bin/activate  # On Unix/MacOS
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Install Ollama and pull a compatible model:
```bash
# Install Ollama from https://ollama.ai/
# Then pull the model
ollama pull llama3.2:3b-instruct-fp16
```

5. Set up your API keys:
```bash
# Get your Tavily API key from https://tavily.com/
# Get your LangSmith API key from https://smith.langchain.com/
# You'll be prompted for these when running the application if not set as environment variables
```

## Usage

### Running the Application

The main entry point is `src/main.py`, which provides an interactive question-answering interface:

```bash
python src\main.py
```

This will:
1. Load the Llama 3.2 model via Ollama
2. Process documents from URLs listed in `rag_urls.txt`
3. Create a vector store for efficient retrieval
4. Set up the graph-based workflow
5. Start an interactive loop where you can ask cybersecurity questions

### Example Questions

The system is particularly well-suited for questions about:
- Exploit techniques and vulnerabilities
- Memory protection mechanisms
- Side-channel attacks
- Kernel security
- Hardware vulnerabilities

Examples:
- "What is a buffer overflow attack and how does it work?"
- "Explain the Rowhammer attack technique"
- "How do ASLR bypass techniques work?"
- "What are the best practices for preventing SQL injection?"

### Customizing Document Sources

The system uses URLs listed in `rag_urls.txt` as document sources. You can modify this file to include your own sources:

```
https://example.com/cybersecurity-doc1
https://example.com/cybersecurity-doc2
```

## System Architecture

CyberRAGLLM uses a graph-based workflow with the following components:

1. **Question Routing**: Determines whether to use vectorstore or web search
2. **Document Retrieval**: Fetches relevant documents from the vectorstore
3. **Document Grading**: Assesses the relevance of retrieved documents
4. **Web Search**: Supplements with web search results when needed
5. **Answer Generation**: Creates answers using RAG with the retrieved documents
6. **Quality Control**: Checks for hallucinations and relevance to the question

## Requirements

- Python 3.8+
- Ollama with a compatible local LLM model (e.g., Llama 3.2)
- Dependencies:
  - langchain, langchain_core, langchain_community
  - langchain_ollama
  - langgraph
  - scikit-learn
  - tiktoken
  - tavily-python
  - bs4 (BeautifulSoup)
- Tavily API key for web search functionality
- LangSmith API key for tracing (optional)
- Sufficient RAM for embedding and running the LLM

## License

This project is licensed under the terms included in the LICENSE file.
