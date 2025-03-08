import streamlit as st


from langchain.chains import create_history_aware_retriever, create_retrieval_chain

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatMessagePromptTemplate,MessagesPlaceholder ,ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os
from dotenv import load_dotenv

load_dotenv()
os.environ['HUGGING_FACE_TOKEN']=os.getenv('HUGGING_FACE_TOKEN')
embeddings=HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')





st.set_page_config(page_title="Conversational RAG with PDF Uploads & Chat History", layout="wide")
st.header("Conversational RAG With PDF uploads and Chat History")

with st.sidebar:
    api_key1=st.text_input('Provied Groq API key',type='password')

    session_id=st.text_input('Session ID',value='default_session')
    uploaded_files=st.file_uploader('Upload Document',accept_multiple_files=True)



api_key=os.getenv('GROQ_API')

if api_key:
    model=ChatGroq(
        groq_api_key=api_key,model_name='Gemma2-9b-It')

    # Chat Interface
    # session_id=st.text_input("Session ID ",value='default_session')
    #Manage the Chat history

    if 'store' not in st.session_state:
        st.session_state.store={}

    # uploaded_files=st.file_uploader("Choose a PDF file",type='pdf',accept_multiple_files=True)

    #processing the file

    if uploaded_files:
        documents=[]

        for uploaded_file in uploaded_files:
            temppdf=f"./temp.pdf"

            with open(temppdf,"wb") as file:
                file.write(uploaded_file.getvalue())

                file_name=uploaded_file.name
            
            loader=PyPDFLoader(temppdf)
            docs=loader.load()
            documents.extend(docs)

        splitter=RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=500)
        splits=splitter.split_documents(documents)

        vectorstore=Chroma.from_documents(documents=splits,embedding=embeddings,persist_directory="./chroma_db")

        retriever=vectorstore.as_retriever()

        contextualize_q_system_prompt=(
       
        "Given a chat history and the latest user question"
        "which might reference context in the chat history, "        
        "formulate a standalone question which can be understood "
        
        "without the chat history. Do NOT answer the question, "
        
        "just reformulate it if needed and otherwise return it as is."
        
    )
        contextualize_q_prompt=ChatPromptTemplate.from_messages(
        [
            ('system',contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ('human','{input}')
        ]
    )

        history_aware_retriever=create_history_aware_retriever(
            model,
           retriever,contextualize_q_prompt)
    
    #Answer Question

        system_prompt=(
            "You are an assistant for question-answering tasks. " 
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. "
            "Don't answer for the questions which is not relevent to your context"
            "\n\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
                   ]
                )

        question_answer_chain=create_stuff_documents_chain(model,qa_prompt)

        rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

        def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]

        conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,get_session_history,input_messages_key='input',
            history_messages_key='chat_history',
           output_messages_key='answer'
        )  

        # userinput=st.text_input("Query : ")
        userinput=st.text_input(' ',placeholder='Ask your Queries...')

        if userinput:
            session_history=get_session_history(session_id)

            response=conversational_rag_chain.invoke(
                {'input':userinput},
                config={
                    'configurable':{'session_id':session_id}
                },
                )
            print(response['answer'])
            # st.write(st.session_state.store)
            st.write(response['answer'])
            # st.write("Chat History : ",session_history.messages)

else:
    
    st.warning('Enter your API Key')