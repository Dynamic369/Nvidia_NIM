import streamlit as st
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings,ChatNVIDIA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()

## load the nvidia api key
os.environ["NVIDIA_API_KEY"] = os.getenv("NVIDIA_API_KEY")

def vector_embedding(pdf_files):
    if 'vectors' not in st.session_state and pdf_files:
        st.session_state.embeddings = NVIDIAEmbeddings()
        all_docs = []
        for uploaded_file in pdf_files:
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
            loader = PyPDFLoader(temp_path)
            docs = loader.load()
            all_docs.extend(docs)
            os.remove(temp_path)  # Clean up temp file
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(all_docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)


st.title("Nvidia NIM Demo")
llm = ChatNVIDIA(model='meta/llama-3.1-70b-instruct')

prompt = ChatPromptTemplate.from_template(
    """ 
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}   
<context>   
Questions: {input}

    """)


question=st.text_input("Enter Your Question From Documents")

# Add a button to clear session state/history
if st.button("Clear Session State"):
    for key in ["embeddings", "text_splitter", "final_documents", "vectors"]:
        if key in st.session_state:
            del st.session_state[key]
    st.success("Session state cleared.")

# Add file uploader for multiple PDFs
pdf_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if st.button('Documents Embedding'):
    if pdf_files:
        vector_embedding(pdf_files)
        st.write("Vector Store DB Is Ready")
    else:
        st.warning("Please upload at least one PDF file first.")


import time

if question:
    if 'vectors' in st.session_state:
        documnet_chain = create_stuff_documents_chain(llm, prompt=prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, documnet_chain)
        start = time.process_time()
        response = retrieval_chain.invoke({"input": question})
        print('Response time:', time.process_time() - start)
        st.write(response['answer'])

        with st.expander("Document similarity search"):
            for i, doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.write("-------------------------------------------")
    else:
        st.warning("Please embed a PDF document first.")
