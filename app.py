import os
# from altair import themes
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback



def main(): 
    load_dotenv()
    
    print(os.getenv('OPENAI_API_KEY'))

    st.set_page_config(page_title='PDF-Assistant', page_icon=':smiley:', layout='centered')
    st.header("Ask your PDF 💭")
    pdf = st.file_uploader("Upload your PDF here" , type = 'pdf')


    #extract the text
    if pdf:
        pdf_reader = PdfReader(pdf)
        text = "".join(page.extract_text() for page in pdf_reader.pages)
            
        
        # st.write(text)
        # ------- It will read pdf -----------

        # ------divide the text into chunks -----
        text_splitter = CharacterTextSplitter(
            separator='\n',
            chunk_size=1000,
            chunk_overlap = 200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # st.write(chunks)
        
        # ---- Create embeddings----
        embeddings = OpenAIEmbeddings()
        Knowledge_base = FAISS.from_texts(chunks , embeddings)
        
        # Show UI
        user_question = st.text_input("Ask your questions about your PDF : ")

        if user_question:
            docs = Knowledge_base.similarity_search(user_question)
            # st.write(docs)
            
            llm = OpenAI()
            
            from langchain.chains.question_answering import load_qa_chain
            
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback as callback: #check price
                response = chain.run(input_documents=docs, question=user_question)
                print(callback) #check price
            
        
            st.write(response)
            
if __name__ == "__main__":
    main()