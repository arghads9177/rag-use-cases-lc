# gradio_rag_app.py

import gradio as gr
import os
import hashlib
from PIL import Image

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import google.generativeai as genai
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from unstructured.partition.pdf import partition_pdf
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import shutil

# Import Libraries
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["HUGGINGFACE_API_KEY"] = os.getenv("HUGGINGFACE_API_KEY")

# Explicitly configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Gemini Vision model
vision_model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest")
prompt_text = """
You are an assistant tasked with summarizing tables and images for retrieval.
These summaries will be embedded and used to retrieve the raw image or raw table elements.
Give a concise summary of the image or table that will be optimized for retrieval.
"""

# Define prompt
template = """You are a helpful assistant for question answering task.
your task is to answer the query of the user from the provided context.
Your answer should be concise and to the point.

Question: {question}
Context: {context}
Answer:
"""

# Embedding model
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
DB_DIR = "chroma_db"
vectorstore = None

# Text model
llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro")

prompt = ChatPromptTemplate.from_template(template)


def clear_app():
    # Remove extracted images folder
    if os.path.exists("./extracted_images"):
        shutil.rmtree("./extracted_images")
    
    # Remove chroma_db folder
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")
    
    # Clear UI elements
    return (
        gr.update(value=None),  # file_input
        gr.update(value=""),    # status_output
        [],                     # chatbot (cleared)
    )


def hash_image(file_path):
    with open(file_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def summarize_image(image_path):
    try:
        img = Image.open(image_path)
        response = vision_model.generate_content([prompt_text, img])
        return response.text
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None
    
def extract_text(pdf_path):
    # Load text from a PDF file
    loader = PyMuPDFLoader(pdf_path)
    text_docs = loader.load()
    # Split the documents into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splitted_text = splitter.split_documents(text_docs)
    return splitted_text

def extract_image_and_table(pdf_path):
    image_summaries = {}
    image_hash_set = set()

    elements = partition_pdf(
            filename=pdf_path,
            strategy="hi_res",
            extract_images_in_pdf=True,
            extract_image_block_types=["Image", "Table"],
            extract_image_block_to_payload=False,
            extract_image_block_output_dir="./extracted_images"
        )
        

    # text_elements = [el for el in elements if el.category in ["NarrativeText", "Title", "List"] and el.text]

    # Hash and remove duplicates
    unique_files = []
    for file_name in os.listdir("./extracted_images"):
        file_path = os.path.join("./extracted_images", file_name)
        if os.path.isfile(file_path):
            img_hash = hash_image(file_path)
            if img_hash not in image_hash_set:
                image_hash_set.add(img_hash)
                unique_files.append(file_path)
            else:
                os.remove(file_path)

    # Summarize images
    for img_path in unique_files:
        summary = summarize_image(img_path)
        if summary:
            image_summaries[img_path] = summary

    print("Images summarized")
    # Build Langchain docs
    image_docs = [
        Document(page_content=summary, metadata={"source": path, "type": "image_or_table"})
        for path, summary in image_summaries.items()
    ]

    return image_docs


def extract_and_embed(pdf_files):
    global vectorstore
    all_docs = []
    

    if os.path.exists("./chroma_db"):
        vectorstore = Chroma(
            persist_directory=DB_DIR,
            embedding_function= embedding
        )
        return "✅ PDFs has already been processed and embedded successfully."

    if not os.path.exists("./extracted_images"):
        os.makedirs("./extracted_images")

    for pdf_path in pdf_files:
        print(f"Processing {pdf_path}")
        text_docs = extract_text(pdf_path)
        print("Text extracted and splitted")

        image_docs = []
        image_docs = extract_image_and_table(pdf_path)
        print("Images and tables extracted")


        all_docs.extend(text_docs + image_docs)
        print("Document processed")

    # Embed and store in Chroma
    vectorstore = Chroma.from_documents(documents=all_docs, embedding=embedding)
    return "✅ PDFs processed and embedded successfully."

def chat_with_docs(message, history):
    if vectorstore is None:
        return "⚠️ Please upload and process PDFs first."

    retriever = vectorstore.as_retriever(top_k=3)
    print(f"Query:\n{message}")
    retrieved_docs = retriever.get_relevant_documents(message)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    print(f"Context:\n{context}")
    chain = (prompt
            | llm
            | StrOutputParser())

    try:
        response = chain.invoke({"question": message, "context": context})
        print(response)
        return history + [(message, response)]
    except Exception as e:
        print(e)
        return history + [(message, f"❌ Error: {str(e)}")]

# Gradio UI
def build_ui():
    with gr.Blocks() as demo:
        gr.Markdown("## 📄 Gemini PDF Q&A Assistant (Text + Images + Tables)")

        with gr.Row():
            file_upload = gr.File(label="Upload PDFs", file_types=['.pdf'], file_count="multiple")
            upload_button = gr.Button("📥 Process PDFs")
            clear_button = gr.Button("🧹 Clear", variant="secondary")

        status_output = gr.Textbox(label="Status")

        chatbot = gr.Chatbot()
        query = gr.Textbox(label="Ask a question about your documents")
        submit = gr.Button("💬 Submit")

        upload_button.click(fn=extract_and_embed, inputs=file_upload, outputs=status_output)
        submit.click(fn=chat_with_docs, inputs=[query, chatbot], outputs=chatbot)
        clear_button.click(fn=clear_app, outputs=[file_upload, status_output, chatbot])

    return demo

app = build_ui()

if __name__ == "__main__":
    app.launch()
