import os
import pickle
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("Please set the GEMINI_API_KEY in the .env file.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)
llm = genai.GenerativeModel("gemini-pro")

# Streamlit UI setup
st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

# Collect URLs from user input
urls = [st.sidebar.text_input(f"URL {i + 1}") for i in range(3)]
process_url_clicked = st.sidebar.button("Process URLs")

# File to store FAISS index
file_path = "faiss_store.pkl"

if process_url_clicked:
    if any(urls):
        try:
            st.info("Loading data from URLs...")
            loader = UnstructuredURLLoader(urls=urls)
            data = loader.load()

            st.info("Splitting text...")
            text_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", ".", ","], chunk_size=1000
            )
            docs = text_splitter.split_documents(data)

            st.info("Creating embeddings and building FAISS index...")
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(docs, embeddings)

            with open(file_path, "wb") as f:
                pickle.dump(vectorstore, f)

            st.success("Processing complete! You can now ask questions.")
        except Exception as e:
            st.error(f"Error processing URLs: {e}")
    else:
        st.warning("Please enter at least one valid URL.")

# User query input
query = st.text_input("Ask a question:")

if query:
    if os.path.exists(file_path):
        try:
            with open(file_path, "rb") as f:
                vectorstore = pickle.load(f)

            retriever = vectorstore.as_retriever()
            docs = retriever.get_relevant_documents(query)

            context = "\n".join([doc.page_content for doc in docs])
            prompt = f"Based on the following content, answer the question:\n\n{context}\n\nQuestion: {query}"

            response = llm.generate_content(prompt)
            answer = response.text

            st.header("Answer")
            st.write(answer)

            st.subheader("Sources:")
            for doc in docs:
                st.write(doc.metadata.get("source", "Unknown source"))
        except Exception as e:
            st.error(f"Error retrieving answer: {e}")
    else:
        st.warning("Please process URLs first before asking questions.")
