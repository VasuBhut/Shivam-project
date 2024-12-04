import os
import streamlit as st
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Path to CSV and FAISS index
CSV_FILE = "preprocessed_sales_data.csv"
FAISS_INDEX_PATH = "faiss_index"

# Load CSV data
@st.cache_data(show_spinner=False)
def load_data_and_create_index():
    loader = CSVLoader(file_path=CSV_FILE)
    documents = loader.load()

    # Initialize HuggingFace embeddings
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create or load FAISS index
    if os.path.exists(FAISS_INDEX_PATH):
        vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings_model, allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_documents(documents, embeddings_model)
        vectorstore.save_local(FAISS_INDEX_PATH)

    return vectorstore

vectorstore = load_data_and_create_index()

# Initialize lightweight LLM
@st.cache_resource(show_spinner=False)
def initialize_generator():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)

generator = initialize_generator()

# Helper functions
def generate_response(generator, question, context):
    """
    Generate an answer by combining context and question.
    """
    input_text = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    response = generator(input_text, max_length=200, num_return_sequences=1)
    return response[0]["generated_text"]

def get_context_from_query(query, retriever):
    """
    Retrieve relevant context from the vectorstore based on the query.
    """
    results = retriever.get_relevant_documents(query)
    context = " ".join([doc.page_content for doc in results])
    return context

# Streamlit UI
st.set_page_config(page_title="Chatbot with RAG", layout="centered")

st.title("📄 Chatbot with Retrieval-Augmented Generation (RAG)")
st.write("Ask questions based on the data!")

user_query = st.text_input("Enter your question:", key="query", placeholder="Type here...")
if user_query:
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    with st.spinner("Generating response..."):
        context = get_context_from_query(user_query, retriever)
        response = generate_response(generator, user_query, context)

    st.markdown(f"*You:* {user_query}")
    st.markdown(f"*Chatbot:* {response}")