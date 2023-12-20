import streamlit as st
from streamlit_chat import message
def add_character(text):
    character = "!"
    return text + character

if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello"]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey"]

response_container = st.container()
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        
        user_input = st.text_input("Query:", placeholder="Ask me questions", key='input')
        submit_button = st.form_submit_button(label='Send')
        
    if submit_button and user_input:
        output = add_character(user_input)

        st.session_state['history'].append(user_input)
        st.session_state['generated'].append(output)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state['history'][i])
            #message(st.session_state["history"][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))
