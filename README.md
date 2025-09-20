# Medical Document Assistant

## Overview

Medical Document Assistant is an intelligent medical document assistant that allows users to upload PDF medical documents and ask questions about their content. The application uses advanced natural language processing and retrieval-augmented generation (RAG) to provide accurate, context-aware answers based on the uploaded documents.

Built with a focus on usability and medical relevance, Medical Document Assistant enables healthcare professionals and patients to quickly extract insights from complex medical reports, prescriptions, lab results, and research papers.

---

## Features

- 📁 **PDF Document Upload**: Easily upload one or multiple medical PDFs for analysis  
- 🔍 **Intelligent Search**: Semantic search powered by FAISS vector database for precise information retrieval  
- 💬 **Interactive Chat Interface**: Natural language Q&A—ask follow-up questions just like a conversation  
- 🏥 **Medical Context Awareness**: Optimized for understanding clinical terminology, diagnoses, medications, and procedures  
- ⚡ **Fast & Efficient Processing**: Lightweight backend with rapid text extraction and indexing using LangChain and PyPDF  

---

## Technology Stack

| Component               | Technology Used                          |
|------------------------|------------------------------------------|
| **Frontend**           | Streamlit                                |
| **Vector Database**    | FAISS (Facebook AI Similarity Search)     |
| **Embedding Model**    | `all-mpnet-base-v2` (HuggingFace)         |
| **Language Model**     | GPT-4.1-nano               |
| **PDF Processing**     | PyPDF                                      |
| **Text Splitting**     | LangChain RecursiveCharacterTextSplitter  |
| **Environment Mgmt**   | Python-dotenv                            |

---

## Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/shaswot84/Medical-Document-Assistant.git
cd Medical-Document-Assistant
```
### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```
### Step 3: Set Up Environment Variables
Create a .env file in the project root:
```bash
echo "API_KEY=your_llm_api_key_here" > .env
```
### Step 4: Usage
Launch the app:
```bash
streamlit run main.py
```
Upload your medical PDF files
Start chatting :) 

Project Structure
``` bash
medichat-pro/
├── app/
│   ├── main.py              # Streamlit entry point
│   ├── ui.py                # Sidebar and file upload components
│   ├── pdf_utils.py         # Extracts and cleans text from PDFs
│   ├── vectorstore_utils.py # Handles FAISS index creation/querying
│   ├── chat_utils.py        # Manages LLM interactions and RAG pipeline
│   └── config.py            # App constants and paths
├── requirements.txt         # Required Python packages
├── .env                     # Environment variables (not tracked)
└── README.md                # This documentation
```




