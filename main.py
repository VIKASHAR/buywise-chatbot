import streamlit as st
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

st.set_page_config(page_title="Document Genie", layout="wide")

# Get the Google API key from the environment variable
api_key = os.getenv("GOOGLE_API_KEY")  # Make sure to set this in your .env file

# Path to the single CSV file
csv_file = "cleaned_amazon.csv"

def get_csv_text(csv_file_path):
    """Get text from a single CSV file."""
    df = pd.read_csv(csv_file_path)
    text = df.to_string(index=False)
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def initialize_vector_store():
    """Initialize vector store from the CSV file if it doesn't already exist."""
    if not os.path.exists("faiss_index"):
        st.info("Creating FAISS vector store from the CSV file...")
        raw_text = get_csv_text(csv_file)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks, api_key)
        st.success("Vector store created successfully!")

def get_conversational_chain():
    prompt_template = """
        As a customer support representative, provide a detailed response based on the customer review context. Your goal is to recommend a product that best fits the customer's needs or preferences highlighted in the review. If the answer is not in the provided context, simply state, "The answer is not available in the context." Please do not provide any incorrect answers.
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

def main():
    st.header("BuyWise AI chatbot")

    # Initialize the vector store if it hasn't been created already
    initialize_vector_store()

    user_question = st.text_input("Ask a Question (eg: give me the review and specification of oneplus tv)", key="user_question")

    if user_question:  # Ensure user question is provided
        user_input(user_question)

if __name__ == "__main__":
    main()
