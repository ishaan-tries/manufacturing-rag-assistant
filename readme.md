# Manufacturing Documentation RAG Assistant

A RAG (Retrieval Augmented Generation) system that answers questions 
from technical manufacturing manuals using LangChain, FAISS, and Google Gemini.

## How it works
1. ingest.py - loads PDF, splits into chunks, creates FAISS vector store
2. rag.py - loads vector store, retrieves relevant chunks, generates answers

## Demo
Question: how do I connect my phone
Answer: To connect your phone, follow these steps:
1. Activate your phone's Bluetooth...
[Sources cited from manual pages]

## Stack
- LangChain LCEL for chain orchestration
- FAISS for vector similarity search  
- Google Gemini for embeddings and generation
- PyPDF for document loading
