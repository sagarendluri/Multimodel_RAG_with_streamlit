# Multimodel_RAG_with_streamlit
Multi_modal_RAG
![image](https://github.com/user-attachments/assets/4e160abf-70f3-474e-b278-f0e7b60d549a)
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
 


Okay, I can definitely help you write the installation documentation for your Multimodal RAG project. Here's a step-by-step guide based on the assumption that your `requirements.txt` contains the necessary Python packages and focusing on installing Ollama and running the specified models.

## Installation Guide for Multimodal RAG with Ollama

This guide will walk you through the steps to install the necessary software and Python packages to run the Multimodal RAG application.

**Prerequisites:**

* **Python 3.6 or higher** is recommended.
* **Git** should be installed to clone the repository.

**Step 1: Clone the Repository**

First, clone the GitHub repository to your local machine:

```bash
git clone https://github.com/sagarendluri/Multimodel_RAG_with_streamlit
cd Multimodel_RAG_with_streamlit
```

**Step 2: Install Python Dependencies**

Navigate to the project directory and install the required Python packages listed in the `requirements.txt` file. It's highly recommended to use a virtual environment to manage dependencies.

**Using `venv` (Recommended):**

```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate  # On Windows
pip install -r requirements.txt
```

**Using `conda` (If you prefer Conda):**

```bash
conda create --name rag_env python=3.13
conda activate rag_env
pip install -r requirements.txt
```

**(Note:** I will now access the `requirements.txt` file from your GitHub link to provide a more specific list of dependencies if needed.)
Unfortunately, I am unable to directly access and view the content of files within GitHub repositories at this time. Therefore, I cannot provide a precise list of the Python packages in your `requirements.txt`.

**However, your `requirements.txt` file likely includes libraries such as:**

* `streamlit`
* `llama-index`
* `langchain` (you mentioned using it previously)
* `transformers`
* `sentence-transformers` (based on the error you encountered)
* `fitz` (or `PyMuPDF`)
* Potentially other libraries for image processing, etc.

**Step 3: Install Ollama**

Ollama is a tool for running large language models locally. You need to install it on your system.

* **macOS:** The easiest way is to download the Ollama application from the official website: [https://ollama.com/download](https://ollama.com/download). Simply download and open the `.dmg` file, then drag the Ollama icon to your Applications folder.

* **Linux:** You can install Ollama using the following command in your terminal:

    ```bash
    curl -fsSL https://ollama.com/install | sh
    ```

* **Windows:** You can download the installer from the official website: [https://ollama.com/download](https://ollama.com/download).

**Step 4: Run Llama3:2 and Gemma 3:4b in Ollama**

Once Ollama is installed, you can download and run the Llama3:2 and Gemma 3:4b models. Open your terminal and run the following commands:

```bash
ollama pull llama3:2
ollama pull gemma:7b-it # Note: You mentioned gemma3:4b, but the available tag is gemma:7b-it or similar. Please check Ollama's model list for the exact tag.
```

**(Important Note on Gemma Model Tag):** The exact tag for the Gemma model might vary. Please visit the Ollama model library ([https://ollama.com/library](https://ollama.com/library)) and search for "gemma" to find the correct tag (it might be `gemma:7b-it`, `gemma:2b`, etc.). Use the appropriate tag in the `ollama pull` command.

**Step 5: Run Your Streamlit Application**

After installing the dependencies and pulling the Ollama models, you can run your Streamlit application from the project directory:

```bash
streamlit run your_streamlit_script_name.py
```

Replace `your_streamlit_script_name.py` with the actual name of your main Streamlit Python file.

**Troubleshooting Tips:**

* **Environment Issues:** If you encounter errors related to missing packages, ensure your virtual environment is activated.
* **Ollama Not Running:** If you have issues connecting to the Ollama models, make sure the Ollama application is running in the background (especially on macOS). On Linux, Ollama runs as a service.
* **Model Names:** Double-check the model names in your code to ensure they match the names you pulled with Ollama (e.g., `llama3:2`, `gemma:7b-it`).
* **Error Messages:** Pay close attention to any error messages in the terminal or in your Streamlit app. These messages often provide clues about what went wrong.

This documentation should help users install the necessary components to run your Multimodal RAG application with Ollama. Remember to replace placeholders like `your_streamlit_script_name.py` with the actual names of your files.
