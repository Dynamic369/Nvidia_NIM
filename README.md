# Nvidia_NIM
## For APP1.PY
# Nvidia NIM PDF Chat Demo

This Streamlit app allows you to upload multiple PDF files, embed their content using NVIDIA's LLM embeddings, and ask questions about the documents in natural language. The app uses LangChain, FAISS for vector storage, and NVIDIA's NIM LLM for answering your queries.

## Features

- **Upload Multiple PDFs:** Select and upload several PDF files at once.
- **Document Embedding:** Embed all uploaded PDFs into a vector database for semantic search.
- **Natural Language Q&A:** Ask questions about your documents and get context-aware answers.
- **Session Management:** Clear embeddings and history with a single click.

## Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd <repo-folder>
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   - Create a `.env` file in the project directory.
   - Add your NVIDIA API key:
     ```
     NVIDIA_API_KEY=your_nvidia_api_key_here
     ```

## Usage

1. **Run the app**
   ```bash
   streamlit run aap1.py
   ```

2. **In the app:**
   - Upload one or more PDF files.
   - Click **Documents Embedding** to process the files.
   - Enter your question in the text box.
   - View the answer and document context.
   - Use **Clear Session State** to reset the app.

## File Structure

- `aap1.py` — Main Streamlit application.
- `.env` — Store your NVIDIA API key here.

## Requirements

- Python 3.8+
- streamlit
- langchain
- langchain_nvidia_ai_endpoints
- langchain_community
- python-dotenv
- faiss-cpu

## FOR APP.PY
# Nvidia NIM PDF Q&A Demo

This Streamlit app enables you to perform question answering over a collection of PDF documents using NVIDIA's LLM and LangChain. It loads PDFs from a directory, embeds their content, and allows you to ask questions in natural language. The app uses FAISS for vector storage and NVIDIA's NIM LLM for generating answers.

---

## Features

- **PDF Directory Loader:** Loads all PDFs from a specified folder (`./us_census` by default).
- **Document Embedding:** Splits and embeds document chunks using NVIDIAEmbeddings.
- **Semantic Search & Q&A:** Ask questions about the documents and get context-aware answers.
- **Document Similarity View:** See the most relevant document chunks for each answer.

---

## How It Works

1. **Load PDFs:** The app uses `PyPDFDirectoryLoader` to load all PDFs from the `./us_census` directory.
2. **Embed Documents:** Documents are split into chunks and embedded using NVIDIA's embedding model.
3. **Store Vectors:** Embeddings are stored in a FAISS vector database for fast retrieval.
4. **Ask Questions:** Enter a question; the app retrieves relevant chunks and uses NVIDIA's LLM to answer.
5. **View Context:** Expand the "Document similarity search" section to see the most relevant document chunks.

---

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd <repo-folder>
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   - Create a `.env` file in the project directory.
   - Add your NVIDIA API key:
     ```
     NVIDIA_API_KEY=your_nvidia_api_key_here
     ```

4. **Add your PDFs**
   - Place your PDF files in the `us_census` directory (or change the path in the code).

5. **Run the app**
   ```bash
   streamlit run app.py
   ```

---

## Main Code Components

- **vector_embedding():** Loads and embeds PDF documents if not already in session state.
- **ChatNVIDIA:** Used as the LLM for answering questions.
- **FAISS:** Stores and retrieves document embeddings for semantic search.
- **Streamlit UI:** Simple interface for embedding documents and asking questions.

---

## Requirements

- Python 3.8+
- streamlit
- langchain
- langchain_nvidia_ai_endpoints
- langchain_community
- python-dotenv
- faiss-cpu

