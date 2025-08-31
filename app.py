import streamlit as st
from src.helper import get_pdf_text, get_text_chunks, get_vector_store, get_conversational_chain
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Function to handle user input
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    # Display response in styled box
    st.markdown(
        f"""
        <div style='padding:20px; border-radius:12px; background-color:#f9f9f9; border:1px solid #ddd;'>
            <h4 style='color:#333;'>💡 Answer:</h4>
            <p style='color:#444; font-size:16px; line-height:1.5;'>{response["output_text"]}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Main function
def main():
    st.set_page_config(page_title="Chat PDF with Gemini", page_icon="📄", layout="wide")

    # Custom CSS for styling
    st.markdown(
        """
        <style>
            body {font-family: 'Segoe UI', sans-serif;}
            .main-header {font-size: 2.8rem; font-weight: 700; text-align: center; color: #2c3e50; margin-top: 10px;}
            .sub-header {font-size: 1.2rem; text-align: center; color: #7f8c8d; margin-bottom: 25px;}
            .stTextInput > div > div > input {border-radius: 10px; border: 1px solid #ccc; padding: 10px;}
            .stButton button {background-color: #2c3e50; color: white; border-radius: 8px; font-weight: 600; padding: 10px 20px;}
            .stButton button:hover {background-color: #34495e;}
            .sidebar .sidebar-content {background: #f8f9fa; border-right: 1px solid #eee;}
        </style>
        """,
        unsafe_allow_html=True
    )

    # Header
    st.markdown("<div class='main-header'>📄 Chat with PDF using Gemini</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Upload PDFs, ask questions, and get AI-powered answers instantly.</div>", unsafe_allow_html=True)

    # User input
    user_question = st.text_input("Ask a Question", placeholder="Type your question here...")
    if user_question:
        with st.spinner("🤖 Generating response..."):
            user_input(user_question)

    # Sidebar
    with st.sidebar:
        st.markdown("## 📂 Upload & Process PDFs")
        st.markdown("<p style='color: #555;'>Upload your PDF files to build a searchable knowledge base.</p>", unsafe_allow_html=True)

        pdf_docs = st.file_uploader("Upload PDF Files", accept_multiple_files=True, type=["pdf"])

        if st.button("🚀 Submit & Process"):
            if pdf_docs:
                with st.spinner("📚 Processing your files..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("✅ Files processed successfully! Now ask your questions.")
            else:
                st.warning("⚠️ Please upload at least one PDF file.")

if __name__ == "__main__":
    main()
