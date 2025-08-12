# gradio_rag_app.py

import gradio as gr
import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import google.generativeai as genai
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Import Libraries
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["HUGGINGFACE_API_KEY"] = os.getenv("HUGGINGFACE_API_KEY")

# Explicitly configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Define prompt
template = """You are a helpful assistant for question answering task.
your task is to answer the query of the user from the provided context.
Your answer should be concise and to the point.

Question: {question}
Context: {context}
Answer:
"""

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
DB_DIR = "chroma_db"
vectorstore = None

llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro")

prompt = ChatPromptTemplate.from_template(template)

# Create retirever
vectorstore = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embedding
)

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
        gr.Markdown("## 📄 Softmeets Buddy")

        # with gr.Row():
        #     upload_button = gr.Button("📥 Process Website")
        #     clear_button = gr.Button("🧹 Clear", variant="secondary")

        status_output = gr.Textbox(label="Status")

        chatbot = gr.Chatbot()
        query = gr.Textbox(label="Ask a question about Softmeets")
        submit = gr.Button("💬 Submit")

        # upload_button.click(fn=extract_and_embed, outputs=status_output)
        submit.click(fn=chat_with_docs, inputs=[query, chatbot], outputs=chatbot)
        # clear_button.click(fn=clear_app, outputs=[status_output, chatbot])

    return demo

app = build_ui()

if __name__ == "__main__":
    app.launch()
