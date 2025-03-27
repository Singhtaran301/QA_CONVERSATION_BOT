import streamlit as st
import os
from typing_extensions import List, TypedDict

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, MessagesState, END, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

# Load API Key
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ API KEY is missing!")
    st.stop()

# Initialize LLM
llm = ChatGroq(model_name="llama3-8b-8192", api_key=groq_api_key)

# Prompt Template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the following context:
    <context>
    {docs}
    </context>

    Question: {question}
    """
)


def create_vector_embedding():
    if "vector_store" not in st.session_state:
        if 'uploaded_file' not in st.session_state:
            st.error("Please upload a file")
            return
        
        uploaded_file = st.session_state.uploaded_file

        file_path = f'temp_{uploaded_file.name}'
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        loader = PyPDFLoader(file_path)
        docs = loader.load()  

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_docs = text_splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.vector_store = FAISS.from_documents(final_docs, embeddings)  
        st.session_state.retriever = st.session_state.vector_store.as_retriever()  

        st.success("Vector DB is ready")


st.title("Conversational RAG with PDF Upload")
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file:
    st.session_state.uploaded_file = uploaded_file  

if st.button("Document Embedding"):
    create_vector_embedding()

user_prompt = st.text_input("Enter your query.......")


class State(TypedDict):
    question: str
    response: str
    documents: List[str]

def retrieve_docs(state: State): 
    if "vector_store" not in st.session_state:
        st.error("Please create vector database first by clicking 'Document Embedding'.")
        return {}

    retriever = st.session_state.retriever
    question = state["question"]  
    docs = retriever.invoke(question)

    return {"documents": docs}


def generate(state: State):  
    if "documents" not in state:  
        st.error("Can't find retrieved documents. Please provide retrieved documents for response generation.")
        return {}

    question = state["question"] 
    docs = state["documents"]  

    rag_chain = prompt | llm | StrOutputParser()  
    response = rag_chain.invoke({"question": question, "docs": docs})  #

    return {"response": response}


graph = StateGraph(State)

graph.add_node("retrieve", retrieve_docs)
graph.add_node("generate", generate)

graph.add_edge(START, "retrieve") 
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)  
app = graph.compile()

if st.button("Run Chat"):
    if not user_prompt:
        st.error("Please enter a query.")
    else:
        output = app.invoke({"question": user_prompt})
        st.write("Response:", output.get("response", "No response generated"))
