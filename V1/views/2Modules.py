import streamlit as st
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader    #changed to community
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import threading
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import ollama_embedding_function
from langchain_community.retrievers import BM25Retriever
from transformers import DistilBertTokenizer, DistilBertModel
import torch

# or
# client = chromadb.HttpClient(url="http://your_chromadb_server")

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')
def create_embeddings(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

if 'collection_name' not in st.session_state:
    st.session_state.collection_name = 'my_collection'

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

if 'module' not in st.session_state:
    with open('modules.txt', 'r') as f:
        modules = f.read().splitlines()
        if len(modules[0]) > 0: 
            st.session_state.module = ''
        else: 
            with open('lastActiveModule.txt', 'r') as f:
                st.session_state.module = f.read()

max_threads = 5
semaphore = threading.Semaphore(max_threads)
threads = []

def get_files_in_path(path):
    return os.listdir(path)

def set_module(module : str):
    st.session_state.module = module
    with open('lastActiveModule.txt', 'w') as f:
        f.write(st.session_state.module)

def get_module():
    with open('lastActiveModule.txt', 'r') as f:
        module = f.read().splitlines()[0]
        st.session_state.module = module
    return module

def get_last_Active_Module():
    with open('lastActiveModule.txt', 'r') as f:
        return f.read().splitlines()

def Create_module(module : str) -> bool:
    dirs = os.listdir()
    for dir in dirs:
        if dir == module:
            return False
    os.mkdir(module)
    os.chdir(module)
    os.mkdir('raw')
    os.mkdir('processed')
    PersistentClient(path=os.getcwd()+'/processed/')
    os.chdir('..')
    with open('modules.txt', 'a') as f:
        f.write(module + '\n')
    st.session_state.path = os.getcwd()
    st.write(st.session_state.path)
    st.session_state.module = module
    return True

def get_PATH():
    st.session_state.PATH = os.getcwd()

#if 'vectorestore' not in st.session_state:
    #st.session_state.vectorestore = Chroma(persist_directory='data',
#def get_data_path() -> str:
    #data_path = os.path.join(get_PATH(),'data','raw', st.session_state.get(['Institute'].copy))
    #return data_path
#    pass
def get_modules():
    with open('modules.txt', 'r') as f:
        return f.read().splitlines()

if 'vectorestore' not in st.session_state:
    st.session_state.vectorestore = None
    #Chroma(persist_directory='data',
    #                    embedding_function=OllamaEmbeddings(
    #                        model="mistral")
    #                    )


def reload():
    st.session_state.reload_trigger = True
    
FILE_PATH = st.session_state.module + '/raw/'
SAVE_DIR = get_module() +'/processed/' 

reload()
def get_data_path():
        PATH = f'{FILE_PATH}'
        return PATH
def read_pdf(file_name):
    with open(file_name, "rb") as pdf_file:
        return pdf_file.read()

def does_file_exist(uploaded_file):
    bytes_data = uploaded_file
    for file in os.listdir(get_module() + '/raw/'):
        fc = open(get_data_path() + file, "rb")
        if fc.read() == bytes_data:
            return True
    f = open(get_module() +'/raw/' + upload_data.name, "wb")
    f.write(bytes_data)
    f.close()
    return False 
# Create a download button for the PDF file

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = True



button_css = """
<style>
div[data-testid="stButton"] button {
    font-size: 20px !important;
    padding: 10px 40px !important;
    border-radius: 15px !important
}
div[data-testid="stFileUploader"] button {
    font-size: 20px !important;
    border-radius: 15px !important
}
div[data-testid="]


p {
    padding: 8px 20px !important;
}
</style>
"""
st.markdown(button_css, unsafe_allow_html=True)
st.markdown(
    "<h1 style='text-align: center;'>Modules</h1>",
    unsafe_allow_html=True
)
with open('modules.txt', 'r') as f:
    modules = f.read().splitlines()

modules_int = 0
new_module = st.text_input('', placeholder='Enter new module name')
if st.button('Create new module'):
    if Create_module(new_module):
        st.switch_page('views/2Modules.py')
    else:
        st.error('Module already exists.')
st.divider()
col1,col2,col3,col4 = st.columns((1,1,1,1),gap='small', vertical_alignment='center')                          
if st.session_state.module != '' and get_data_path != '':
    column1 ,column2, column3 = st.columns((1,2,1), vertical_alignment='center')
    st.write('Your modules')
    for module in get_modules():
        modules_int += 1
        if modules_int == 1:
            with col1:
                if st.button(module):
                    set_module(module)
        if modules_int == 2:
            with col2:
                if st.button(module):
                    set_module(module)
        if modules_int == 3:
            with col3:
                if st.button(module):
                    set_module(module)
        if modules_int == 4:
            with col4:
                if st.button(module):
                    set_module(module)
            modules_int = 0           
    with st.expander('Files for ' + get_module(), expanded=False, icon=':material/arrow_drop_down:'):
        files = os.listdir(get_module() + '/raw/')
        if len(files) == 0:
            st.write("No files found for " + get_module())
        else:
            for file in files:
                st.write(file)

                
                    
else:
    st.write('You currently have no modules. Please add a new module to get started.')
    module = st.text_input('',placeholder='Enter module name')
    if st.button('Add new module'):
        if Create_module(module):
            st.switch_page('views/2Modules.py')
        else:
            st.error('Module already exists.')
         
upload_data = st.file_uploader(f'Upload PDF for {get_module()}', type=['pdf'], accept_multiple_files=True)
colm1 ,colm2, colm3 = st.columns((2,1,2)) 
 
with colm1:
    if st.button('Choose files you want to upload and chat with'):
        for file in upload_data:
            st.write(upload_data.name)
            if not os.path.isfile(get_module() +'/raw/' + upload_data.name):
                with st.status("Analyzing your document..."):
            #ORC to find alike documents
                    bytes_data = upload_data.read()
            #measure by binary size and store extra info if new one extentds other
                    st.error('before function')
                    if not does_file_exist(bytes_data):
                        st.success('No similar file found. Uploading new file')
                        st.success('files bytes have been uploaded')
                #try:
                        loader = PyPDFLoader(file_path= get_module() +'/raw/' + upload_data.name)     #+".pdf"
                        data = loader.load()
                        st.success("File uploaded successfully")
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=400,
                            chunk_overlap=50,
                            length_function=len
                            )
                        st.write('text splitter over')
                        all_splits = text_splitter.split_documents(data)
                        client = PersistentClient(path=get_module() +'/processed/chromadb.sqlite3')
                        collection = client.get_or_create_collection(
                            name="my_collection",
                            embedding_function=OllamaEmbeddings(model="mistral"),
                            
                        )
                        SAVE_DIR = get_module() +'/processed/' 
                        for split in all_splits:
                            embeddings = create_embeddings(split)
                            collection.add(documents=split, embeddings=embeddings)
                            
                    with open('vectorstores.txt', 'a') as f:
                        pass
                 
                #except Exception as e:
                    #os.chdir(get_module() +'/raw/')
                    #os.remove(filename.name)
                    #os.chdir(f'{st.session_state.PATH}')
                #    st.error('Something went wrong. Please try again.')
                #for i, split in enumerate(all_splits):
                #    thread = threading.Thread(target=embed_chunk, args=(split,  i, semaphore))
                #    threads.append(thread)
                #    thread.start()

                #for thread in threads:
                #    thread.join()

            # Create and persist the vector store
    
                    

            # Create and persist the vector store
    
            else:
                st.info('File already exists but thanks for trying! ')   
        #st.status("Done!")
        
        st.info('finished')
#with colm3:
#    if st.button('Upload files'):
#        st.write(upload_data.name)
#        if not os.path.isfile(get_module() +'/raw/' + upload_data.name):
#            with st.status("Analyzing your document..."):
 #           #ORC to find alike documents
#                bytes_data = upload_data.read()
#            #meaure by binary size and store extra info if new one extentds other
#                st.error('before function')
#                if not does_file_exist(bytes_data):
#                    st.success('No similar file found. Uploading new file')
#                    st.success('files bytes have been uploaded')
        
#colm1 ,colm2, colm3 = st.columns((1,2,1))
#with colm2:
#    st.write('Choose multiple files for AI to process')
#saved_files = get_files_in_path(get_module() +'/raw/')
#mod_files = []
#for file in saved_files:
#    mod_files.append(file)
#selected_files = st.multiselect("Select files", mod_files)
#with colm2:
#    if st.button('Process selected files'):
#        docs = []
#        for file in selected_files:
#            loader = PyPDFLoader(file_path= get_module() +'/raw/' + file)     #+".pdf"
#            docs.extend(loader.load())
#        text_splitter = RecursiveCharacterTextSplitter(
#                    chunk_size=1500,
#                    chunk_overlap=200,
#                    length_function=len
#                    )
#        with st.status("Creating embeddings..."):
#            embedding = OllamaEmbeddings(model="mistral")
#            all_splits = text_splitter.split_documents(docs)
#            vectorstore = Chroma.from_documents(
#                    documents=all_splits,
#                    embedding=embedding,
#                    persist_directory=get_module() +'/processed/',
#                    )
    

            
            #loader = PyPDFLoader("files/"+get_current_Module() +'/' + uploaded_file.name+".pdf")     #+".pdf"
            #data = loader.load()

            # Initialize text splitter
            #text_splitter = RecursiveCharacterTextSplitter(
            #    chunk_size=1500,
            #    chunk_overlap=200,
            #    length_function=len
            #)
            #all_splits = text_splitter.split_documents(data)

            # Create and persist the vector store
            #st.session_state.vectorstore = Chroma.from_documents(
            #    documents=all_splits,
            #    embedding=OllamaEmbeddings(model="mistral")
            #)
            #st.session_state.vectorstore.persist()    


