import tabula
import base64
import pymupdf
import os
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
import ollama
from pathlib import Path

# Create the directories
def create_directories(base_dir):
    directories = ["images", "text", "image_text" ,"tables", "page_images"]
    for dir in directories:
        os.makedirs(os.path.join(base_dir, dir), exist_ok=True)

# Process tables
def process_tables(filepath, doc, page_num, base_dir, items):
    try:
        tables = tabula.read_pdf(filepath, pages=page_num + 1, multiple_tables=True)
        if not tables:
            return
        for table_idx, table in enumerate(tables):
            table_text = "\n".join([" | ".join(map(str, row)) for row in table.values])
            table_file_name = f"{base_dir}/tables/{os.path.basename(filepath[:-4])}_table_{page_num}_{table_idx}.txt"
            with open(table_file_name, 'w') as f:
                f.write(table_text)
            items.append({"page": page_num, "type": "table", "text": table_text, "path": table_file_name})
    except Exception as e:
        print(f"Error extracting tables from page {page_num}: {str(e)}")
# Process text chunks
def process_text_chunks(filepath,text, text_splitter, page_num, base_dir, items):
    chunks = text_splitter.split_text(text)
    for i, chunk in enumerate(chunks):
        text_file_name = f"{base_dir}/text/{os.path.basename(filepath[:-4])}_text_{page_num}_{i}.txt"
        with open(text_file_name, 'w') as f:
            f.write(chunk)
        items.append({"page": page_num, "type": "text", "text": chunk, "path": text_file_name})

# Process images
def process_images(filepath,doc, page, page_num, base_dir, items):
    images_path = []
    images = page.get_images()
    for idx, image in enumerate(images):
        xref = image[0]
        pix = pymupdf.Pixmap(doc, xref)
        image_name = f"{base_dir}/images/{os.path.basename(filepath[:-4])}_image_{page_num}_{idx}_{xref}.png"
        pix.save(image_name)
        with open(image_name, 'rb') as f:
            encoded_image = base64.b64encode(f.read()).decode('utf8')
        
        images_path.append(image_name)
        items.append({"page": page_num, "type": "image", "path": image_name, "image": encoded_image})
    return images_path

# Process page images
def process_page_images(filepath, page, page_num, base_dir, items):
    pix = page.get_pixmap()
    page_path = os.path.join(base_dir, f"page_images/page_{page_num:03d}.png")
    pix.save(page_path)
    with open(page_path, 'rb') as f:
        page_image = base64.b64encode(f.read()).decode('utf8')
    items.append({"page": page_num, "type": "page", "path": page_path, "image": page_image})

def images_to_text(images_path, base_dir,filepath,page_num):

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=200, length_function=len)

    for image in images_path:
        try:
            response = ollama.chat(
                model='gemma3:4b',
                messages=[
                    {
                        'role': 'user',
                        'content': 'Describe this image',
                        'images': [image]
                    }
                ]
            )
            print(response)

            text_content = response['message']['content']
            chunks = text_splitter.split_text(text_content)

            for i, chunk in enumerate(chunks):
                text_file_name = f"{base_dir}/image_text/{os.path.basename(filepath[:-4])}_text_{page_num}_{i}.txt"

                with open(text_file_name, 'w') as f:
                    f.write(chunk)

        except Exception as e:
            print(f"Error processing image {image}: {e}")

def list_files_with_extension(folder_path, extension):
    folder_path = Path(folder_path)
    if not folder_path.exists() or not folder_path.is_dir():
        return []

    file_lists = [file.name for file in folder_path.glob(f"*{extension}")]
    return file_lists

def process_pdfs(folder_path: str,base_dir:str):
    print("folder_path",folder_path)
    
    extension = ".pdf"
    pdf_files = list_files_with_extension(folder_path, extension)
    print("pdf_files",pdf_files)
   
    items = []

    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path,pdf_file)
        print("pdf_path",pdf_path)
        try:
            doc = pymupdf.open(pdf_path)
            num_pages = len(doc)
            

            create_directories(base_dir)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, length_function=len)

            for page_num in tqdm(range(num_pages), desc="Processing PDF pages"):
                page = doc[page_num]
                text = page.get_text()
                process_tables(pdf_path, doc, page_num, base_dir, items)
                process_text_chunks(pdf_path, text, text_splitter, page_num, base_dir, items)
                images_path = process_images(pdf_path,doc, page, page_num, base_dir, items)
                images_to_text(images_path,base_dir,pdf_path,page_num)
                process_page_images(pdf_path, page, page_num, base_dir, items)
            doc.close()

        except Exception as e:
            print(f"Error processing PDF file {pdf_path}: {e}")