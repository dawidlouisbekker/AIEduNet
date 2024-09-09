from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.llms import Ollama
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import streamlit as st
import os
import time



if not os.path.exists('files'):
    os.mkdir('files')

if not os.path.exists('data'):
    os.mkdir('data')

if 'template' not in st.session_state:
    st.session_state.template = """You are a knowledgeable chatbot, here to help with questions of the user. Your tone should be professional and informative.

    Context: {context}
    History: {history}

    User: {question}
    Chatbot:"""
if 'prompt' not in st.session_state:
    st.session_state.prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=st.session_state.template,
    )
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True,
        input_key="question")
    #datadirectory

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = Chroma(persist_directory= os.getcwd() + st.session_state.module + '/processed/',
                        embedding_function=OllamaEmbeddings(
                            model="mistral")
                        )
else:
    st.session_state.vectorstore = Chroma(persist_directory= os.getcwd() + st.session_state.module + '/processed/',
                        embedding_function=OllamaEmbeddings(
                            model="mistral")
                        )
    
if 'llm' not in st.session_state:
    st.session_state.llm = Ollama(base_url="http://localhost:11434",
                model="mistral",
                verbose=True,
                callback_manager=CallbackManager(
                    [StreamingStdOutCallbackHandler()]),
                )

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

st.title(st.session_state.module + " PDF Chatbot")
st.write(os.getcwd() + '/' + st.session_state.module + '/processed/')
st.write(st.session_state.vectorstore)

#documents = st.multiselect("Select Documents", os.listdir(), key="documents")
# Upload a PDF file
#uploaded_powerpoint = st.file_uploader('Upload you power point', type=['ppt', 'pptx'])

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["message"])
    
def set_vectorstore(module):
    for db in os.listdir(os.getcwd() + '/' + st.session_state.module + '/processed/'):
        pass


st.session_state.retriever = st.session_state.vectorstore.as_retriever()
    # Initialize the QA chain
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = RetrievalQA.from_chain_type(
        llm=st.session_state.llm,
        chain_type='stuff',
        retriever=st.session_state.retriever,
        verbose=True,
        chain_type_kwargs={
            "verbose": True,
            "prompt": st.session_state.prompt,
            "memory": st.session_state.memory,
            }
        )

    # Chat input
if user_input := st.chat_input("You:", key="user_input"):
    user_message = {"role": "user", "message": user_input}
    st.session_state.chat_history.append(user_message)
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Assistant is typing..."):
            response = st.session_state.qa_chain(user_input)
        message_placeholder = st.empty()
        full_response = ""
        for chunk in response['result'].split():
            full_response += chunk + " "
            time.sleep(0.05)
                # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    chatbot_message = {"role": "assistant", "message": response['result']}
    st.session_state.chat_history.append(chatbot_message)
