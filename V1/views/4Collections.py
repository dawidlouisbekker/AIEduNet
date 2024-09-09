import streamlit as st 
from transformers import DistilBertTokenizer, DistilBertModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
import chromadb
from chromadb import PersistentClient
from langchain_community.document_loaders import PyPDFLoader
import os
import time

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')
def create_embeddings(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

def set_last_module(module : str):
    with open('lastActiveModule.txt', 'w') as f:
        f.write(module)
        f.close()

if 'share' not in st.session_state:
    st.session_state.share = False


if 'files' not in st.session_state:
    st.session_state.files = ''

def set_files_from_module(module : str):
    st.session_state.files = os.listdir(os.getcwd() + '/' + module + '/raw/')

def set_files_from_files_dir():
    pass

if 'collection_name' not in st.session_state:
    st.session_state.collection_name = ''



Collections_PATH = 'collections/' 

st.title("collections")


collection_name = st.text_input('Add new collection', placeholder='Enter collection name')
if st.session_state.files == '':
    st.subheader('Select module first ot upload files')
    collection_files = st.multiselect('Select a module below', None)
else:
    files = st.session_state.files
    collection_files = st.multiselect('Select files', files)
    if st.session_state.internet:
        help = st.checkbox('Check if others can help with embeddings', key='help_with_embeddings')
    else:
        st.write('Connect to the internet so other computers can help you process embeddings')
    if st.checkbox('Share your collection with others?', key='share_collection', value=False):
        st.session_state.share = True 
   
    if st.button('Create new collection'):
        save_PATH = Collections_PATH + collection_name + '/'
        os.mkdir(save_PATH)
        collection_metadata = {
        'collection_name': collection_name,
        'description': 'This collection contains embeddings of PDF documents.',
        'creation_date': '',
    # Add any other metadata fields you need
}
        for file in files:
            if help:
                pass
            loader = PyPDFLoader(file_path= st.session_state.module + '/raw/' + file.name)     #+".pdf"
            data = loader.load()
            st.success("File uploaded successfully")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=400,
                chunk_overlap=50,
                length_function=len
                )
            all_splits = text_splitter.split_documents(data)
            client = PersistentClient(path= os.getcwd()'/collections/chromadb.sqlite3')
            collection = client.get_or_create_collection(
            name=collection_name,)
            for split in all_splits:
                embeddings = create_embeddings(split)
                metadata = {'file_name': file.name}
                collection.add(documents=split, embeddings=embeddings)
        
        
    
    

st.divider()
modules_int = 0
col1,col2,col3,col4 = st.columns((1,1,1,1),gap='small', vertical_alignment='center')  


for module in st.session_state.modules:
    modules_int += 1
    if modules_int == 1:
        with col1:
            if st.button(module):
                st.session_state.modules = module
                set_last_module(module)
                set_files_from_module(module)
                st.switch_page('views/4Collections.py')
    if modules_int == 2:
        with col2:
            if st.button(module):
                st.session_state.modules = module
                set_last_module(module)
                set_files_from_module(module)
                st.switch_page('views/4Collections.py')
    if modules_int == 3:
        with col3:
            if st.button(module):
                st.session_state.modules = module
                set_last_module(module)
                set_files_from_module(module)
                st.switch_page('views/4Collections.py')
    if modules_int == 4:
        with col4:
            if st.button(module):
                st.session_state.modules = module
                set_last_module(module)
                set_files_from_module(module)
                st.switch_page('views/4Collections.py')
            modules_int = 0  
st.divider()

#if st.session_state.internet:
#    st.checkbox('Check if others can help with embeddings', key='help_with_embeddings')
#else:
#    st.write('Connect to the internet so other computers can help you process embeddings')
    

    
    