import streamlit as st
import os
import cv2
import numpy as np
import easyocr
from PIL import Image
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_groq import ChatGroq

# التأكد من المفتاح
if "GROQ_API_KEY" in st.secrets:
    api_key = st.secrets["GROQ_API_KEY"]
else:
    st.error("يرجى إضافة GROQ_API_KEY في Secrets")
    st.stop()

@st.cache_resource
def load_system():
    reader = easyocr.Reader(['en'])
    pdf_path = "Reference.pdf" 
    if os.path.exists(pdf_path):
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(split_docs, embeddings)
        llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile", temperature=0.1)
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=vector_store.as_retriever())
        return reader, qa
    return reader, None

reader, qa = load_system()

st.title("🩺 MedBot Pro")
up_file = st.file_uploader("ارفع صورة الروشتة", type=['jpg','png','jpeg'])

if up_file:
    img = Image.open(up_file)
    st.image(img, use_container_width=True)
    if st.button("تحليل الآن 🚀"):
        with st.spinner("جاري التحليل..."):
            img_array = np.array(img)
            res = reader.readtext(img_array, detail=0)
            raw_text = " ".join(res)
            query = f"Identify drugs in: {raw_text}. Provide clinical counseling in Egyptian Arabic based on the reference."
            response = qa.run(query)
            st.success(response)
