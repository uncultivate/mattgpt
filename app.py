import os
import tempfile
import logging
from typing import List
import streamlit as st
from streamlit_chat import message
from rag import ChatPDF

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

st.set_page_config(page_title="Literature Review AI", layout="wide")

# Initialize session state variables
def init_session_state():
    default_values = {
        'active_tab': "qa",
        'messages': {},
        'pdf_count': 0,
        'assistant': ChatPDF(),
    }
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

def reset_application():
    st.session_state.assistant.clear()
    st.session_state.messages = {}
    st.session_state.pdf_count = 0
    st.session_state.active_tab = "qa"

def display_messages(tab: str):
    for i, (msg, is_user) in enumerate(st.session_state.messages.get(tab, [])):
        message(msg, is_user=is_user, key=f"{tab}_{i}")

def process_input(tab_identifier: str):
    input_key = f"user_input_{tab_identifier}"
    user_input = st.session_state.get(input_key, "")

    if user_input:
        messages = st.session_state.messages.setdefault(tab_identifier, [])
        with st.spinner("Thinking..."):
            try:
                agent_text = st.session_state.assistant.ask(user_input, tab_identifier)
                messages.extend([(user_input, True), (agent_text, False)])
            except ValueError as e:
                messages.extend([(user_input, True), (str(e), False)])
                logging.warning(f"Error processing input: {e}")

        st.session_state.messages[tab_identifier] = messages
        st.session_state[input_key] = ""
        st.session_state.active_tab = tab_identifier

def read_and_save_file():
    uploaded_files = st.session_state.get("file_uploader")
    if uploaded_files:
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
                tf.write(file.getbuffer())
                file_path = tf.name

            with st.spinner(f"Ingesting {file.name}"):
                st.session_state.pdf_count += 1
                st.session_state.assistant.ingest(file_path, st.session_state.pdf_count)

            os.remove(file_path)
            logging.info(f'File {file.name} processed and removed.')

def main():
    st.header("Literature Review AI")

    # Sidebar
    with st.sidebar:
        st.image('cq.webp', width=150)
        st.subheader('Version: 0.12')
        if st.button("Clear Chat"):
            reset_application()

    # File upload
    st.subheader("Upload Documents")
    st.file_uploader("Upload PDF documents", type=["pdf"], key="file_uploader", 
                     on_change=read_and_save_file, label_visibility="collapsed", 
                     accept_multiple_files=True)
    st.write(f"Total PDFs uploaded: {st.session_state.pdf_count}")

    # Tabs
    tab_mapping = {
        "qa": "Q & A",
        "category_search": "Category Search",
    }
    tabs = st.tabs(list(tab_mapping.values()))

    for identifier, title in tab_mapping.items():
        with tabs[list(tab_mapping.keys()).index(identifier)]:
            st.subheader(title)
            display_messages(identifier)
            st.text_input(f"Enter your {title} query",
                          key=f"user_input_{identifier}",
                          on_change=process_input,
                          args=(identifier,))

if __name__ == "__main__":
    main()