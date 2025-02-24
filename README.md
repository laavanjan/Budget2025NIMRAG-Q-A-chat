# 🇱🇰 Budget Speech 2025 Q&A Chatbot with NVIDIA NIM & LangChain 🤖

This project is a Retrieval Augmented Generation (RAG) based Question & Answer chatbot designed to provide insights into Sri Lanka's Budget Speech 2025. It leverages NVIDIA NIM for powerful language model processing and LangChain for seamless integration and workflow management. 🚀

## Overview

This application allows users to ask questions related to the Sri Lanka Budget Speech 2025 documents. The system uses NVIDIA embeddings and FAISS for vector storage, combined with the NVIDIA's `meta/llama3-70b-instruct` model to generate accurate and context-aware responses. 📚

## Features

-   **Document Ingestion:** Loads PDF documents from a specified directory (`./pdf_fol`).
-   **Text Chunking:** Splits documents into manageable chunks using `RecursiveCharacterTextSplitter`.
-   **Vector Embeddings:** Generates embeddings using NVIDIA Embeddings and stores them in a FAISS vector store.
-   **Question Answering:** Uses LangChain's retrieval chain to answer user questions based on the document context.
-   **Contextual Responses:** Provides responses grounded in the provided budget speech documents.
-   **Document Similarity Search:** Displays relevant document chunks used to generate the response.
-   **Streamlit UI:** User-friendly web interface for easy interaction.
-   **Performance Tracking:** Measures and displays response time.

## Technologies Used

-   **Python:** Core programming language.
-   **Streamlit:** For building the interactive web interface.
-   **LangChain:** Framework for developing applications powered by language models.
-   **NVIDIA NIM:** For powerful language model processing and embeddings (`meta/llama3-70b-instruct`).
-   **FAISS:** For efficient similarity search of embeddings.
-   **PyPDFDirectoryLoader:** Document loader for PDF files.
-   **dotenv:** For managing environment variables.
-   **Anaconda:** For virtual environment management.

## Getting Started

### Prerequisites

-   Python 3.7+ (Recommended Python 3.9+)
-   Anaconda installed.
-   NVIDIA API Key (set as environment variable `NVIDIA_API_KEY`)
-   LangChain API Key (set as environment variable `LANGCHAIN_API_KEY`)
-   LangChain Project Name (set as environment variable `LANGCHAIN_PROJECT`)
-   PDF documents of Sri Lanka's Budget Speech 2025 in the `./pdf_fol` directory.

### Installation

1.  Clone the repository:

    ```bash
    git clone [https://github.com/laavanjan/Budget2025NIMRAG-Q-A-chat.git](https://www.google.com/search?q=https://github.com/laavanjan/Budget2025NIMRAG-Q-A-chat.git)
    cd Budget2025NIMRAG-Q-A-chat
    ```

2.  Create an Anaconda virtual environment:

    ```bash
    conda create -p rag_env python=3.9 -y
    conda activate rag_env
    ```

3.  Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

4.  Place your PDF documents in the `./pdf_fol` directory.

5.  Set your environment variables in a `.env` file (or directly in your environment).

### Running the Application

```bash
streamlit run app.py
