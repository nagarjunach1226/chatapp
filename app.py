from dotenv import load_dotenv
import os
import streamlit as st
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain




def main():
    load_dotenv()
    st.set_page_config(page_title="Ask yout PDF")
    st.header("Ask your PDF ")
    
    pdf = st.file_uploader("upload your PDF ",type="pdf")
    
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text=""
        
        for page in pdf_reader.pages:
            text += page.extract_text()
        
    text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 100,
    chunk_overlap  = 20,
    length_function = len,
    
    )
        
    chunks=text_splitter.split_text(text)
    #st.write(chunks)
        
    #Embeddings
    
    embeddings=OpenAIEmbeddings()
    Vectorstore = FAISS.from_texts(chunks,embedding=embeddings)
    store_name = pdf.name[:4]
    if os.path.exists(f"{store_name}.pkl"):
        
        with open(f"{store_name}.pkl", "rb") as f:
            pickle.load(f)
        
    else:
        
        with open(f"{store_name}.pkl", "wb") as f:
            pickle.dump(Vectorstore,f)
    query=st.text_input("Enter the query")
    
    if query:
        docs= Vectorstore.similarity_search(query=query,k=3)
        llm=OpenAI()
        chain=load_qa_chain(llm,chain_type="stuff")
        response=chain.run(input_documents=docs,question=query)
        st.write(response)        
    
  
if __name__ == "__main__":
    main()
