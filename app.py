import gradio as gr
import os
import cv2
import numpy as np
import easyocr
from PIL import Image
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_groq import ChatGroq

# إعداد المفتاح
api_key = os.getenv("GROQ_API_KEY")

def process_meds(image, question):
    reader = easyocr.Reader(['en'])
    pdf_path = "reference.pdf"
    
    # استخراج النص من الصورة
    img_array = np.array(image)
    res = reader.readtext(img_array, detail=0)
    raw_text = " ".join(res)
    
    # تجهيز المرجع
    if os.path.exists(pdf_path):
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(split_docs, embeddings)
        llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile", temperature=0.1)
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=vector_store.as_retriever())
        
        query = f"Identify drugs in: {raw_text}. Question: {question}. Provide clinical counseling in Egyptian Arabic."
        return qa.run(query)
    return "ملف المرجع غير موجود!"

# واجهة Gradio
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🩺 MedBot Pro")
    with gr.Row():
        img_input = gr.Image(type="pil", label="ارفع صورة الروشتة")
        txt_input = gr.Textbox(label="سؤال إضافي (اختياري)", placeholder="مثلاً: هل يتعارض مع السكر؟")
    btn = gr.Button("تحليل الروشتة الآن 🚀")
    output = gr.Textbox(label="النتيجة والنصيحة الطبية")
    
    btn.click(fn=process_meds, inputs=[img_input, txt_input], outputs=output)

demo.launch()
