import streamlit as st
import requests

def get_modules():
    with open('modules.txt', 'r') as f:
        return f.read().splitlines()
    
def check_internet(url='http://www.google.com/', timeout=5):
    try:
        response = requests.get(url, timeout=timeout)
        return True
    except (requests.ConnectionError, requests.Timeout):
        return False

st.session_state.internet = check_internet()

st.session_state.modules = get_modules()

        

if 'Serving' not in st.session_state:
    st.session_state.Serving = True

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if 'module' not in st.session_state:
    with open('modules.txt', 'r') as f:
        modules = f.read().splitlines()
        if len(modules) == 0: 
            st.session_state.module = ''
        else: 
            with open('lastActiveModule.txt', 'r') as f:
                st.session_state.module = modules[0]
else: 
    with open('lastActiveModule.txt', 'r') as f:
        st.session_state.module = f.read().splitlines()
    
#For AWS lambda
#from fastapi import FastAPI
#from mangum import Mangum
#app = FastAPI()
#handler = Mangum(app)    
#if __name__ == '__main__':
#    port = 80
#    print(f'Running the FastAPI server on port {port}')
#    uvicorn.run(app, host='0.0.0.0', port=port)



if st.session_state.module != '':
    Modules_page = st.Page(
    page='views/2Modules.py',
    title='Your Modules and Files',
    icon=':material/extension:',
    default=False,
    )
else:

    Modules_page = st.Page(
    page='views/2Modules.py',
    title='Modules',
    icon=':material/extension:',
    default=True,
    )



Home_page = st.Page(
    page='views/1Home.py',
    title='Home',
    icon=':material/account_circle:',
)

ChatBot_page = st.Page(
    page='views/3ChatBot.py',
    title='ChatBot',
    icon=':material/chat:',
    default=False,
)

Collections_page = st.Page(
    page='views/4Collections.py',
    title='Collections',
    icon=':material/collections:',
    default=False,
)

pg =st.navigation(
    {
    'local' : [Home_page, Modules_page, ChatBot_page, Collections_page],
    'Online': [] 
})


pg.run()
#run
