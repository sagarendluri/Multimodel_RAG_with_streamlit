from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage, SystemMessage
from typing import List
from graphviz import Source
import os
import time
import uuid

import torch

class ChatBot:
    def __init__(self, faiss_text_index_path: str, model_name: str = "llama3.2"):
        """
        Initializes the ChatBot with a FAISS index path and Ollama model.

        Args:
            faiss_text_index_path (str): The path to the FAISS text index.
            model_name (str, optional): The Ollama model to use. Defaults to "llama3.2".
        """
        self.faiss_text_index_path = faiss_text_index_path
        print("faiss_model",self.faiss_text_index_path)
        self.model_name = model_name
        self.text_embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
        self.loaded_text_store = FAISS.load_local(
            self.faiss_text_index_path , self.text_embeddings, allow_dangerous_deserialization=True
        )

    def retrieval(self, question: str) -> str:
        """
        Retrieves relevant information and generates a response using the Ollama model.

        Args:
            question (str): The user's question.

        Returns:
            str: The generated response.
        """
        try:
            results = self.loaded_text_store.similarity_search(question)
            context = "\n".join([res.page_content for res in results])
            print("Retrieved Context:", context)

            llm = Ollama(model=f"{self.model_name}:latest", temperature=0)
            negotiate_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant for question answering. The text context is relevant information retrieved."),
                ("user", "{question}"),
                ("user", "{context}")
            ])

            formatted_prompt = negotiate_prompt.format(question=question, context=context)
            response = llm.invoke(formatted_prompt)
            print("Generated Response:", response)

            return response #if hasattr(response, "content") else "Error: Failed to generate response."

        except Exception as e:
            return f"An error occurred: {e}"

    def generate_regulatory_dot_graphviz(self, question: str, context: str) -> str:
        """
        Generates Dot Graphviz code for regulatory requirements based on user question and context.

        Args:
            question (str): The user's question about regulatory requirements.
            context (str): The context containing regulatory information.

        Returns:
            str: The generated Dot Graphviz code or an error message.
        """
        dot_format = """ digraph G {
            // Start node
            start [shape=ellipse, label="Start"];
            // ... (rest of the dot_format code) ...
            } """

        few_shot_examples = [
            {"input": "Give me the rule and exceptions for the regulation...", "output": dot_format}
        ]

        few_shot_template = ChatPromptTemplate.from_messages([
            ("human", "{input}"),
            ("ai", "{output}")
        ])

        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=few_shot_template,
            examples=few_shot_examples,
        )

        negotiate_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a business analyst..."),
            few_shot_prompt,
            ("user", "{question}"),
            ("user", "{context}")
        ])

        try:
            llm = Ollama(model=self.model_name)
            formatted_prompt = negotiate_prompt.format(question=question, context=context)
            response = llm.invoke(formatted_prompt)

            

            return response.content if hasattr(response, "content") else "Error: Failed to generate response."

        except Exception as e:
            return f"An error occurred: {e}"
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
