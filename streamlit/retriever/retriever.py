from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
import os
import torch


# Fix for torch issue
torch.classes.__path__ = [os.path.join(torch.__path__[0], 'torch_classes')]
def ask_question_from_text_index(question: str, faiss_text_index_path: str ):
    """
    Takes a question as input, loads a pre-existing FAISS index for text,
    sets up a retrieval-based question answering chain using Ollama with the llama3.2 model,
    and returns the answer to the question.

    Args:
        question: The question to ask.
        faiss_text_index_path: The path to the saved FAISS text index.

    Returns:
        The answer to the question from the retrieval QA chain.
    """


    faiss_text_index_path = "EU_faiss_text_index.faiss"
    text_embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

    try:
        loaded_text_store = FAISS.load_local(faiss_text_index_path, text_embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        return f"Error loading FAISS index: {e}"

    qa_template = """Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}

    Only return the helpful answer below and nothing else.
    Helpful answer:
    """
    prompt = PromptTemplate(template=qa_template,
                            input_variables=['context', 'question'])

    try:
        llm = Ollama(model="llama3.2")
    except ImportError:
        return "Error: Ollama not found or not running. Please ensure Ollama is installed and running."
    except Exception as e:
        return f"Error initializing Ollama: {e}"

    text_retriever = loaded_text_store.as_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=text_retriever,
        chain_type_kwargs={'prompt': prompt})
    
   
    response = qa_chain.invoke({"query": question})
    print("response,",response)
    return response
    # try:
    #     response = qa_chain.invoke({"query": question})
    #     return response
    
        
    # except Exception as e:
    #     return f"Error during query execution: {e}"

# # Example usage:
# if __name__ == "__main__":
#     user_question = "What are the key aspects of the project charter?"
#     answer = ask_question_from_text_index(user_question)
#     print(f"Question: {user_question}")
#     print(f"Answer: {answer}")