# gradio_rag_app.py

import gradio as gr
import os
import io
import base64
import hashlib

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import pymupdf

import numpy as np
from langchain_community.vectorstores import FAISS

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableMap
import shutil

# Import Libraries
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["HUGGINGFACE_API_KEY"] = os.getenv("HUGGINGFACE_API_KEY")

# Initialize CLIP model for unified embeddings
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Define prompt
template = """You are a helpful assistant for question answering task.
your task is to answer the query of the user from the provided context.
Your answer should be concise and to the point.

Question: {question}
Context: {context}
Answer:
"""

vectorstore = None

# Text model
llm = ChatOpenAI(model="gpt-4.1")

prompt = ChatPromptTemplate.from_template(template)

# Storage for all documents and embeddings
all_docs = []
all_embeddings = []
image_data_store = {}  # Store actual image data for LLM

input_text_folder = "extracted_text_pages"  # Folder of text files
input_pdf_folder = "pdf-docs"          # Folder containing PDF files


def clear_app():
    # Remove extracted images folder
    # if os.path.exists("./extracted_images"):
    #     shutil.rmtree("./extracted_images")
    
    # Remove FAISS folder
    if os.path.exists("./faiss_index"):
        shutil.rmtree("./faiss_index")

    all_docs = []
    all_embeddings = []
    image_data_store = {}
    
    # Clear UI elements
    return (
        gr.update(value=None),  # file_input
        gr.update(value=""),    # status_output
        [],                     # chatbot (cleared)
    )

# Define image embedding function
def embed_image(image_data):
    """Embed image using CLIP"""
    if isinstance(image_data, str):  # If path
        image = Image.open(image_data).convert("RGB")
    else:  # If PIL Image
        image = image_data
    
    inputs=clip_processor(images=image,return_tensors="pt")
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
        # Normalize embeddings to unit vector
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()

# Define text embedding function    
def embed_text(text):
    """Embed text using CLIP."""
    inputs = clip_processor(
        text=text, 
        return_tensors="pt", 
        padding=True,
        truncation=True,
        max_length=77  # CLIP's max token length
    )
    with torch.no_grad():
        features = clip_model.get_text_features(**inputs)
        # Normalize embeddings
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()

# Define image extraction and embeding function
def extract_embed_image(doc):
    """Extract images from a PDF and embed the images"""
    # Text splitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    # Process each page
    for i, page in enumerate(doc):
        ## process text
        text=page.get_text()
        if text.strip():
            ##create temporary document for splitting
            temp_doc = Document(page_content=text, metadata={"page": i, "type": "text"})
            text_chunks = splitter.split_documents([temp_doc])

            #Embed each chunk using CLIP
            for chunk in text_chunks:
                if chunk.page_content.strip():
                    embedding = embed_text(chunk.page_content)
                    all_embeddings.append(embedding)
                    all_docs.append(chunk)
        ## process images
        ##Three Important Actions:

        ##Convert PDF image to PIL format
        ##Store as base64 for GPT-4V (which needs base64 images)
        ##Create CLIP embedding for retrieval

        for img_index, img in enumerate(page.get_images(full=True)):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Convert to PIL Image
                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                
                # Create unique identifier
                image_id = f"page_{i}_img_{img_index}"
                
                # Store image as base64 for later use with GPT-4V
                buffered = io.BytesIO()
                pil_image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                image_data_store[image_id] = img_base64
                
                # Embed image using CLIP
                embedding = embed_image(pil_image)
                all_embeddings.append(embedding)
                
                # Create document for image
                image_doc = Document(
                    page_content=f"[Image: {image_id}]",
                    metadata={"page": i, "type": "image", "image_id": image_id}
                )
                all_docs.append(image_doc)
                
            except Exception as e:
                print(f"Error processing image {img_index} on page {i}: {e}")
                continue

# # Define text loading from from text files and embedding
# def load_embed_text():
#     """Load each text file of a directory, split and embed"""
#     loader = DirectoryLoader(path=input_text_folder, glob="**/*.txt")
#     documents = loader.load()
#     # Text splitter
#     splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
#     text_chunks = splitter.split_documents(documents)

#     #Embed each chunk using CLIP
#     for i, chunk in enumerate(text_chunks):
#         embedding = embed_text(chunk.page_content)
#         chunk.metadata["type"]= "text"
#         chunk.metadata["page"] = i
#         print(chunk.metadata)
#         all_embeddings.append(embedding)
#         all_docs.append(chunk)

def store_embeddings():
    global vectorstore  

    if os.path.exists("./faiss_index"):
        vectorstore = FAISS.load_local("faiss_index", embeddings=None, allow_dangerous_deserialization=True)
        return "✅ Vectors loaded successfully."

    # Process iamges of PDF
    # Process each PDF in the input folder
    for filename in os.listdir(input_pdf_folder):
        pdf_path = os.path.join(input_pdf_folder, filename)
        if filename.endswith(".pdf"):
            try:
                doc = pymupdf.open(pdf_path)
                print(f"Processing: {filename} ({doc.page_count} pages)")
                extract_embed_image(doc)
                doc.close()

            except Exception as e:
                print(f"Failed to process {filename}: {e}")
    # Process text
    # load_embed_text()
    # Create embedding array
    embeddings_array = np.array(all_embeddings)
    # Create custom FAISS index since we have precomputed embeddings
    vectorstore = FAISS.from_embeddings(
        text_embeddings=[(doc.page_content, emb) for doc, emb in zip(all_docs, embeddings_array)],
        embedding=None,  # We're using precomputed embeddings
        metadatas=[doc.metadata for doc in all_docs]
    )
    vectorstore.save_local("./faiss_index")
    return "✅ PDFs processed and embedded successfully."

# Create Retriever function
def retrieve_multimodal(query, k=5):
    """Unified retrieval using CLIP embeddings for both text and images."""
    # Embed query using CLIP
    query_embedding = embed_text(query)
    
    # Search in unified vector store
    results = vectorstore.similarity_search_by_vector(
        embedding=query_embedding,
        k=k
    )
    
    return results

def create_multimodal_message(query, retrieved_docs):
    """Create a message with both text and images for GPT-4V."""
    content = []
    
    # Add the query
    # content.append({
    #     "type": "text",
    #     "text": f"Question: {query}\n\nContext:\n"
    # })
    content.append(
        {
            "type": "text",
            "text": f"You are a helpful assistant for question answering task.Your task is to answer the query of the user from the provided context.Your answer should be concise and to the point.\n\nQuestion: {query}\n\nContext:\n"
        }
    )
    
    # Separate text and image documents
    text_docs = [doc for doc in retrieved_docs if doc.metadata.get("type") == "text"]
    image_docs = [doc for doc in retrieved_docs if doc.metadata.get("type") == "image"]
    
    # Add text context
    if text_docs:
        text_context = "\n\n".join([
            f"[Page {doc.metadata['page']}]: {doc.page_content}"
            for doc in text_docs
        ])
        content.append({
            "type": "text",
            "text": f"Text excerpts:\n{text_context}\n"
        })
    
    # Add images
    for doc in image_docs:
        image_id = doc.metadata.get("image_id")
        if image_id and image_id in image_data_store:
            content.append({
                "type": "text",
                "text": f"\n[Image from page {doc.metadata['page']}]:\n"
            })
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_data_store[image_id]}"
                }
            })
    
    # Add instruction
    # content.append({
    #     "type": "text",
    #     "text": "\n\nPlease answer the question based on the provided text and images."
    # })
    
    return HumanMessage(content=content)

def multimodal_pdf_rag_pipeline_with_chain(query):
    """Main pipeline for multimodal RAG."""
    # Create the RAG chain using Runnable components
    rag_chain = (
        RunnableLambda(lambda query: {"query": query, "docs": retrieve_multimodal(query, k= 5)})
        | RunnableLambda(lambda inputs: {
            "query": inputs["query"],
            "message": create_multimodal_message(inputs["query"], inputs["docs"])
        })
        | RunnableLambda(lambda inputs: llm.invoke([inputs["message"]]))
        | StrOutputParser()
    )
    
    # Get response from GPT-4V
    response = rag_chain.invoke(query)
    
    # Also print retrieved docs
    context_docs = retrieve_multimodal(query)
    # Print retrieved context info
    print(f"\nRetrieved {len(context_docs)} documents:")
    for doc in context_docs:
        doc_type = doc.metadata.get("type", "unknown")
        page = doc.metadata.get("page", "?")
        if doc_type == "text":
            preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            print(f"  - Text from page {page}: {preview}")
        else:
            print(f"  - Image from page {page}")
    print("\n")
    
    return response

def chat_with_docs(message, history):
    if vectorstore is None:
        return "⚠️ Please upload and process PDFs first."
    try:
        response = multimodal_pdf_rag_pipeline_with_chain(message)
        # print(response)
        return history + [(message, response)]
    except Exception as e:
        print(e)
        return history + [(message, f"❌ Error: {str(e)}")]

# Gradio UI
def build_ui():
    with gr.Blocks() as demo:
        gr.Markdown("## 📄 Multimodal PDF Q&A Assistant (Text + Images + Tables)")

        with gr.Row():
            # file_upload = gr.File(label="Upload PDFs", file_types=['.pdf'], file_count="multiple")
            upload_button = gr.Button("📥 Process PDFs")
            clear_button = gr.Button("🧹 Clear", variant="secondary")

        status_output = gr.Textbox(label="Status")

        chatbot = gr.Chatbot()
        query = gr.Textbox(label="Ask a question about your documents")
        submit = gr.Button("💬 Submit")

        upload_button.click(fn=store_embeddings, outputs=status_output)
        submit.click(fn=chat_with_docs, inputs=[query, chatbot], outputs=chatbot)
        clear_button.click(fn=clear_app, outputs=[status_output, chatbot])

    return demo

app = build_ui()

if __name__ == "__main__":
    app.launch()
