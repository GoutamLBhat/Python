import streamlit as st
from PyPDF2 import PdfReader
from langchain.chains.openai_functions.openapi import openapi_spec_to_openai_fn
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from  langchain_community.chat_models import  ChatOpenAI


api_Key="Your Api-Key"
#read Pdf File use streamlit which containes information related to incident management
st.header("Incident Manager")
with st.sidebar:
    st.title("Upload Document to Train Incident LLM")
    file=st.file_uploader("upload pdf file",type="pdf")

#extract the text use pypdf2
if file is not None:
    pdf_Reader = PdfReader(file)
    text=""
    for page in pdf_Reader.pages:
        text+=page.extract_text()
        # st.write(text)

#and breaking into chunks to work on the small part of the test
    #idetifier to break text
    text_Splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len

    )

    chunks= text_Splitter.split_text(text)

 #Create Embedding mean convert Text into Numbers why because system cant read text.
    embedding = OpenAIEmbeddings(openai_api_key=api_Key)
    #vector store to store Embeddings Faiss from face book  will store your text end embeddings
    vector_store = FAISS.from_texts(chunks,embedding)

    # take input question from user

    user_Input = st.text_input("Enter Your Question")

#similarity check compare  with vector db and user input semantic search
    if user_Input:
        match=vector_store.similarity_search(user_Input)


    # show refined out put using open ai
    #define llm  fine_tuning out put can done here
        llm=ChatOpenAI(
            openai_api_key=api_Key,
            temperature=0,
            max_tokens=200,
            model_name="gpt-3.5-turbo"
        )
    #chain -- take a question --get relevant document --pass it to llm --generate out put
        chain=load_qa_chain(llm,chain_type="stuff")
        response = chain.run(input_documents=match,questino=user_Input)
        st.write(response)

