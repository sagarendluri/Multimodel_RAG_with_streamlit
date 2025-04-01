import streamlit as st
import os
import torch
from Ollama.LLM import Ollama  # Ensure this path is correct
from extraction.pdf_processing import process_pdfs
from embedding.embeddings import process_documents_and_create_faiss_index
# from Ollama.pdf_txt_retriever import answer_question
from Ollama.pdf_txt_retriever import ChatBot 
# Fix for torch issue
torch.classes.__path__ = [os.path.join(torch.__path__[0], 'torch_classes')]

st.title("WireFrame Project Interface")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False  # Track whether the PDF is processed
if 'faiss_index_created' not in st.session_state:
    st.session_state.faiss_index_created = False  # Track whether FAISS index is created
if 'faiss_index_path' not in st.session_state:
    st.session_state.faiss_index_path = None # Track the faiss index path.

uploaded_files = st.file_uploader("Attach a file (txt, pdf, docx)", type=["txt", "pdf", "docx"], accept_multiple_files=True)

if uploaded_files and not st.session_state.pdf_processed:
    save_path = "./pdf_files"
    base_dir = "media"
    os.makedirs("media", exist_ok=True)
    os.makedirs(save_path, exist_ok=True)  # Ensure directory exists

    st.write("You have uploaded the following files:")

    for uploaded_file in uploaded_files:
        st.write(f"- {uploaded_file.name}")  # Display file names
        pdf_path = os.path.join(save_path, uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())  # Save file directly

    # st.session_state.pdf_processed = True  # Prevent re-processing on rerun
    # faiss_text_index_path = uploaded_files[0].name[:-4] + "_" + "faiss_text_index.faiss"
    # faiss_text_index_path = "advanced_pdf_1_faiss_text_index.faiss"
    # process_pdfs(save_path, base_dir)
    # st.session_state.pdf_processed = True  # Mark PDF as processed
    # st.session_state.faiss_index_path = process_documents_and_create_faiss_index(base_dir, faiss_text_index_path)
    # st.session_state.faiss_index_created = True  # Mark FAISS index as created
faiss_text_index_path = "saved_advanced_pdf_1_faiss_text_index.faiss"
user_input = st.chat_input("Enter your message...")
# faiss_text_index_path = "advanced_pdf_1_EU_faiss_text_index.faiss"
if user_input:
    st.session_state.chat_history.append({"sender": "user", "message": user_input})
    ollama_instance = ChatBot(faiss_text_index_path) 

    # Check if the user has asked for a wireframe using dot graphviz
    if "wireframe" in user_input.lower() and "dot graphviz" in user_input.lower():
        #Use the faiss index path
        response = ollama_instance.generate_regulatory_dot_graphviz(prompt=user_input)

        if isinstance(response, str):  # Ensure response is a valid string
            image_path = ollama_instance.dot_to_image(dot_code=response)
            
            bot_response = f"Here is the wireframe generated using Dot Graphviz:\n![Wireframe]({image_path})" if image_path else "Failed to generate the wireframe image."
        else:
            bot_response = "Error: Unexpected response format from Ollama."

    else:
        # if st.session_state.faiss_index_path:
        bot_response = ollama_instance.retrieval(user_input)
        # if hasattr(bot_response, "content"):
        #     return bot_response 
        # else:
        #     bot_response = "Please upload a file first."

    st.session_state.chat_history.append({"sender": "bot", "message": bot_response})

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["sender"]):
        st.write(message["message"])