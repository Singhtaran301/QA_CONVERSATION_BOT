import streamlit as st
import langchain
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPdfLoader
from langchain_core.runnable.history import RunnableWithMessageHistory
import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


st.title("Conversational RAG with PDF upload")
st.write("Upload PDF and chat with their content")

llm=ChatGroq(groq_api_key="GROQ_API_KEY",model_name="Gemma-9b-It")

session_id=st.text_input("Session Id",value="default_session")

if 'store' not in st.session_state:
    st.session_state.store={}

uploaded_files=st.file_uploader("Choose A Pdf file", type="pdf",accept_multiple_files=False)

