import os
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
import torch

# Fix for torch issue
torch.classes.__path__ = [os.path.join(torch.__path__[0], 'torch_classes')]

def process_documents_and_create_faiss_index(data_path, faiss_text_index_path):
    """
    Processes documents from a directory, splits them into chunks, and creates a FAISS index.

    Args:
        data_path (str): The path to the directory containing documents.
        faiss_text_index_path (str): The path to save the FAISS index.

    Returns:
        str: The path to the saved FAISS index.
    """

    text_embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

    all_documents = []

    # Load Text Documents Manually (Ensuring file closure)
    for root, _, files in os.walk(data_path):
        for file in files:
            file_path = os.path.join(root, file)
            metadata = {"source": file_path}
            if file_path.endswith('.txt'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        all_documents.append(Document(page_content=content, metadata=metadata))
                except UnicodeDecodeError:
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()
                            all_documents.append(Document(page_content=content, metadata=metadata))
                    except Exception as e:
                        print(f"Error reading text file {file_path}: {e}")
                except Exception as e:
                    print(f"Error reading text file {file_path}: {e}")

    # Split the documents into smaller chunks
    # text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10) #Adjusted chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10, length_function=len)

    docs = text_splitter.split_documents(all_documents)
    print("all_documents",all_documents)

    print(f"Number of text chunks created: {len(docs)}")

    # Create FAISS vector store for text documents
    text_store = FAISS.from_documents(
        docs,
        text_embeddings,
    )

    print(f"Text documents loaded and indexed into FAISS.")

    # Save the FAISS index to disk
    FAISS.save_local(text_store, faiss_text_index_path)
    print(f"FAISS index for text saved to: {faiss_text_index_path}")

    return faiss_text_index_path

# Example Usage:
# data_path = "/path/to/your/data/directory"
# faiss_text_index_path = "/path/to/save/faiss/index.faiss"
# process_documents_and_create_faiss_index(data_path, faiss_text_index_path)