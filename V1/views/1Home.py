import streamlit as st 
import json
import os


if 'logged_in' not in st.session_state:
    st.session_state.logged_in = True


def get_PATH():
    return os.getcwd()

def reload():
    st.query_params(reload=True)


# Display appropriate page based on login state

st.title("Welcome to the Homepage!")
st.write('Nothing here for this version yet!')
st.write('Your collections')