# Multimodel_RAG_with_streamlit
Multi_modal_RAG

This document demonstrates how to implement a multi-modal Retrieval-Augmented Generation (RAG) system using Ollama with Llama3:2 and gemma3:4b and LangChain. This pdfs documents contain a mixture of content types, including tables and images. Traditional RAG applications often lose valuable information captured in images. With the emergence of Multimodal Large Language Models (MLLMs), we can now leverage both text and image data in our RAG systems.
In this notebook, we'll explore one approach to multi-modal RAG (Option 1):
1.	Use multimodal embeddings huggingface to embed both images and text
2.	Retrieve relevant information using similarity search
3.	Pass raw images (Describe the image) and text chunks to a multimodal LLM for answer synthesis using 
We'll use the following tools and technologies:
•	LangChain to build a multimodal RAG system
•	faiss for similarity search
•	Lamma3:2b for answer synthesis
•	Huggingface model for data embeddings
•	pymupdf to parse images, text, and tables from documents (PDFs)
This approach allows us to create a more comprehensive RAG system that can understand and utilize both textual and visual information from our documents.
Prerequisites
Before write the code 	, ensure you have the following packages and dependencies installed:
•	Python 3.13
•	langchain
•	faiss
•	pymupdf
•	tabula
•	tesseract
•	requests
Let's get started with building our multi-modal RAG system using opensource models!
 

