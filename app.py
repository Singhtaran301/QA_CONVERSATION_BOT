import streamlit as st
import langchain

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader

from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, MessagesState
from langgraph.graph import END,START

import os
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from typing_extensions import List,TypedDict
from langgraph.checkpoint.memory import MemorySaver

from dotenv import load_dotenv

load_dotenv()
groq_api_key=os.getenv("GROQ_API_KEY")
#groq_api_key=st.secrets.get("GROQ_API_KEY",os.getenv("GROQ_API_KEY"))

if not groq_api_key:
    st.error("GROQ API KEY is missing!")
    st.stop()

llm=ChatGroq(model_name="llama3-8b-8192",api_key=groq_api_key)

prompt=ChatPromptTemplate.from_template(
    """
    Answer the question based on the following context:
    <context>
    {docs}
    </context>

    Question:{question}
    """
)

def create_vector_embedding():
    if "vectors" not in st.session_state:
        if 'uploaded_file' not in st.session_state:
            st.error("Please upload the file")
            return
        
        uploaded_file=st.session_state.uploaded_file

        file_path=f'temp_{uploaded_file.name}'
        with open(file_path,"wb") as f:
            f.write(uploaded_file.getbuffer())
        
        loader=PyPdfLoader(file_path)
        docs=loader.load

        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        final_docs=text_splitter.split_documents(docs)

        embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.vector_store=FAISS(final_docs,embeddings)
        st.session_state.retriever=vector_store.as_retriever()

        st.success("Vector DB is ready")


st.title("Conversational RAG with PDF upload")
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file:
    st.session_state.uploaded_file = uploaded_file  # Store file persistently

if st.button("Document Embedding"):
    create_vector_embedding()

user_prompt = st.text_input("Enter your query.......")

class State(TypedDict):
    question:str
    response:str
    documents:List[str]


def retrieve_docs(State):

    if question:
        if "vector_store" not in st.session_state:
            st.error("Please create vector database first by clicking 'Document Embedding'.")
        else:
            
            question=state[question]
            docs=retriever.invoke(question)

            return {"documents":docs}

def generate(State):
    if "docs" not in st.session_state:
        st.error("Can't find for retrieved documents. Please provide with the retrieved documents for response generation")
    else:

        question=state[question]
        docs=retriever.invoke(question)

        rag_chain=prompt|llm|StrOutputParser()

        response=rag_chain.invoke(question,docs)


        return {"response":response}

graph=StateGraph(State)

graph.add_edge(START,retrieve)
graph.add_node("retrieve",retrieve_docs)
graph.add_node("generte",generate)
graph.add_edge(generate,END)

app=graph.compile()


        
