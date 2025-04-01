from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from graphviz import Source
import os
import time
import uuid

import torch


# Fix for torch issue
torch.classes.__path__ = [os.path.join(torch.__path__[0], 'torch_classes')]
class FAISSRetriever:
    def __init__(self,index_path):
        self.index_path = index_path
        faiss_text_index_path = "EU_faiss_text_index.faiss"
        text_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        
        try:
            self.loaded_text_store = FAISS.load_local(faiss_text_index_path, text_embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            self.loaded_text_store = None

    def retrieve(self, query):
        if self.loaded_text_store:
            # k = 3
            results = self.loaded_text_store.similarity_search(query)#, k=k)
            context = "\n".join([res.page_content for res in results])
            print("Retrieved Context:", context)
            llm = ChatOllama(model="llama3.2:latest", temperature=0)
            self.negotiate_prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a business analyst with extensive knowledge of export documents and regulatory documentation.Your task is to create a use-case model that accurately captures regulatory requirements and compliance obligations from a given context.Begin by thoroughly understanding the rules and exceptions, ensuring no detail is overlooked. Read every article and paragraph meticulously, focusing on key elements such as goods, technology, manufacture,  entities, and authorization requirements, while also considering any restrictions mentioned. Pay special attention to Annex I and II, especially if they contain multiple paragraphs, to ensure no critical details are missed.Distinguish clearly between prohibitions and obligations to maintain accuracy. Ensure every line is considered, avoiding any omissions.Finally, translate your findings into a detailed digraph G format."),
                
                    ("user", "{question}"),
                    ("user", "{context}")
                ])
            # context = retrieve(query)
            formatted_prompt = self.negotiate_prompt.format(question=query, context=context)
            response = llm.invoke(formatted_prompt )
            print("Generated Response:", response)

            if hasattr(response, "content"):
                return response.content
            return "Error: Failed to generate response."

class Ollama:
    def __init__(self, index_path, model_name="llama3.2:latest", temperature=0):
        """Initialize Ollama with FAISSRetriever and Llama3.2 model."""
        self.retriever = FAISSRetriever(index_path)
        self.llm = ChatOllama(model=model_name, temperature=temperature)

    
    negotiate_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert graph generator. Your task is to generate DOT format digraph code only when the user's request explicitly asks for a graph. If the request is not related to graph generation, provide a clear, concise, and informative text-based response. Do not respond with 'I cannot generate this' or similar vague statements—always provide a meaningful answer."),
        ("user", "{question}"),
        ("user", "{context}")
    ])
    
    # @staticmethod
    def chat(self,prompt):
        
        context = self.retriever.retrieve(prompt)
        # print("context",context)
        formatted_prompt = Ollama.negotiate_prompt.format(question=prompt, context=context)
        response = self.llm.invoke(formatted_prompt)
        print("response",response)

        
        if hasattr(response, "content"):
            return response.content
        return "Error: Failed to generate response."

    # @staticmethod
    def dot_to_image(self, dot_code):
        try:
            unique_id = uuid.uuid4()
            timestamp = int(time.time())
            filename = f"graph_{timestamp}"
            image_dir = "generated_images"
            os.makedirs(image_dir, exist_ok=True)
            image_path = os.path.join(image_dir, f"{filename}.png")
            
            start_index = dot_code.find("digraph ")
            end_index = dot_code.rfind("}")  # Ensures the last closing brace is found
            
            if start_index != -1 and end_index != -1:
                dot_code = dot_code[start_index : end_index + 1]
                graph = Source(dot_code)
                graph.render(directory=image_dir, filename=filename, format="png", cleanup=True)
                return image_path if os.path.exists(image_path) else "Error: Image generation failed."
            return "Error: DOT code not found in the response."
        except Exception as e:
            return f"Graphviz error: {e}"
