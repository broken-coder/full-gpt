# -*- coding: utf-8 -*-
# RAG (Stuff Documents 체인, 수동 구현) + ConversationBufferMemory + 로컬 파일 + 캐시 임베딩

import os
from dotenv import load_dotenv

# ----- 배운 임포트 우선 -----
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler

# StuffDocumentsChain (구 버전 경로)
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import LLMChain

import streamlit as st


st.set_page_config(page_title="DocumentGPT", page_icon="🏠")

class ChatCallbackHandler(BaseCallbackHandler):
    message = ""
    def on_llm_start(self, *args, **kwargs):
       self.message_box = st.empty()
    def on_llm_end(self, *args, **kwargs):
       save_message(self.message, "ai")
      
    def on_llm_new_token(self, token: str, *args, **kwargs):
       self.message += token
       self.message_box.markdown(self.message + "▌")


llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[ChatCallbackHandler()],
)


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/file/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore("./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader("./files/document.txt")
    docs = loader.load_and_split(text_splitter=splitter)

    # ===== 3) 임베딩 + 캐시 임베딩 + 벡터스토어 =====
    base_embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(base_embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

def save_message(message, role):
     st.session_state["messages"].append({"message": message,"role": role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)
       

def paint_messages():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

prompt = ChatPromptTemplate.from_messages([
    ("system", 
     """If you don't know the answer, just say that you don't know, don't try to make up an answer.
     Context: {context}
     """
     ),
    ("human", "{question}"),
   
])

st.title("DocumentGPT")

st.markdown(
    """
Welcome!

Use this chatbot to ask questions to an AI about your files!

upload a file in the sidebar to get started.
"""
)

with st.sidebar:
    file = st.file_uploader("Upload a file", type=["pdf", "docx", "txt"])

if file:
    retriever = embed_file(file)
    send_message("I`m ready! Ask me anything","ai", save=False)
    paint_messages()

    message = st.chat_input("Ask a question about your file:")

    if message:
        send_message(message,"human")
        chain = (
            {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        } | prompt | llm
        )
        with st.chat_message("ai"):
            response = chain.invoke(message)
       
else:
    st.session_state["messages"] = []



