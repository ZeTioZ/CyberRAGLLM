# CyberRAGLLM

CyberRAGLLM is a cybersecurity-focused question-answering system that uses Retrieval-Augmented Generation (RAG) with local LLMs to provide accurate and contextually relevant responses to cybersecurity questions. It leverages a graph-based workflow to combine document retrieval, web search, and quality control mechanisms.

## Features

- 🔒 **Cybersecurity Focus**: Specialized for cybersecurity domain with extensive document collection
- 🤖 **Local LLM Integration**: Works with Ollama-based local LLMs (e.g., Llama 3.2)
- 📚 **Web Document Processing**: Loads and processes documents from web URLs
- 📄 **PDF Support**: Processes PDF files from both local paths and internet URLs
- 🔍 **Vector Search**: Efficient retrieval using SKLearnVectorStore with HuggingFace embeddings
- 🌐 **Web Search Integration**: Uses Tavily Search API for supplementary information
- 🔘 **Web Search Toggle**: Ability to enable or disable web searches completely
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
ollama pull hf.co/safe049/mistral-v0.3-7b-cybersecurity:latest
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
https://example.com/document.pdf
/path/to/local/document.pdf
```

### Using PDF Files

The system can process PDF files from both local paths and internet URLs:

1. **Local PDF Files**: Add the full path to the PDF file in `rag_urls.txt`
2. **Internet PDF Files**: Add the URL to the PDF file in `rag_urls.txt`

The system automatically detects PDF files based on file extension (.pdf) or content type and processes them accordingly.

### Controlling Web Search

You can enable or disable web searches completely:

1. **API Usage**: When using the API, include the `web_search_enabled` parameter in your request:
   ```json
   {
      "model": "your_model_name",
      "messages": [
          { "role": "system", "content": "system_prompt" },
          { "role": "user", "content": "user_prompt" }
      ],
      "web_search_enabled": False,
      "max_retries": 5,
      "temperature": 0.7,
      "max_tokens": -1,
      "stream": False
   }
   ```

2. **Benefits of Disabling Web Search**:
	- Privacy and security when dealing with sensitive information
	- Offline mode for environments without internet access
	- Controlled information to ensure answers are based only on vetted documents
	- Testing and evaluation to compare answer quality with and without web search

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
	- beautifulsoup4 (BeautifulSoup)
	- pypdf (for PDF processing)
- Tavily API key for web search functionality
- LangSmith API key for tracing (optional)
- Sufficient RAM for embedding and running the LLM

## License

This project is licensed under the terms included in the LICENSE file.
