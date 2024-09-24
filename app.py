import os
import tempfile
import logging
from responses import *

import streamlit as st
from streamlit_chat import message
from rag import ChatPDF

# Configure basic logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

st.set_page_config(page_title="CQ")


if 'active_tab' not in st.session_state:
    st.session_state['active_tab'] = "qa" 
if 'messages' not in st.session_state:
    st.session_state['messages'] = {}
if 'pdf_count' not in st.session_state:
    st.session_state.pdf_count = 0  # Initialize count in session state

def reset_application():
    # Clear the ChatPDF instance
    if "assistant" in st.session_state:
        st.session_state["assistant"].clear()
    
    # Reset other relevant session state variables
    st.session_state["messages"] = {}
    st.session_state["user_input"] = {}


def display_messages(tab):
    for i, (msg, is_user) in enumerate(st.session_state["messages"].get(tab, [])):
        message(msg, is_user=is_user, key=f"{tab}_{i}")
    st.session_state["thinking_spinner"] = st.empty()


def process_input(tab_identifier):
    # Retrieve the user input based on the tab_identifier
    input_key = "user_input_qa" if tab_identifier == "qa" else "user_input_category_search"
    user_input = st.session_state.get(input_key, "")

    # Ensure there is input to process and that the assistant has been initialized in the session state
    if user_input and "assistant" in st.session_state:
        # Retrieve or initialize the message list for the current tab
        messages = st.session_state["messages"].setdefault(tab_identifier, [])

        # Display thinking spinner while processing
        with st.spinner(f"Thinking..."):
            try:
                # Call the 'ask' method with the user input and tab_identifier as the query type
                agent_text = st.session_state["assistant"].ask(user_input, tab_identifier)

                # Append both user input and response to the messages list for the current tab
                messages.append((user_input, True))  # True indicating user's message
                messages.append((agent_text, False))  # False indicating system's response
                
            except ValueError:
                messages.append((user_input, True))  # True indicating user's message
                messages.append((not_found, False))  # False indicating system's response
                logging.warning(not_found)

        # Update the session state with the new messages
        st.session_state["messages"][tab_identifier] = messages

        # Optionally, clear the input field after processing
        st.session_state[input_key] = ""
        
        st.session_state['active_tab'] = tab_identifier  # Update the active tab based on the input source



def read_and_save_file():
    st.session_state["assistant"].clear()
    st.session_state["messages"] = {}
    st.session_state["user_input"] = {}

    for file in st.session_state["file_uploader"]:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}"):
            st.session_state.pdf_count += 1
            st.session_state["assistant"].ingest(file_path, st.session_state.pdf_count)
            

        os.remove(file_path)
        logging.info(f'File {file.name} uploaded.')



def page():
    # Individual initialization for session state variables
    if "messages" not in st.session_state:
        st.session_state["messages"] = {}
    if "user_input" not in st.session_state:
        st.session_state["user_input"] = {}
    if "assistant" not in st.session_state:
        st.session_state["assistant"] = ChatPDF()
    if "active_tab" not in st.session_state:
        st.session_state["active_tab"] = "qa"  # Default to the Q & A tab

    st.header("Literature Review AI")

    # Upload document section
    st.subheader("Upload a document")
    st.file_uploader("Upload document", type=["pdf"], key="file_uploader", on_change=read_and_save_file, label_visibility="collapsed", accept_multiple_files=True)
    # Version and reset
    st.sidebar.image('cq.webp', width=150)
    st.sidebar.subheader('Version: 0.12')
    if st.sidebar.button("Clear Chat"):
        reset_application()
    st.write(f"Total PDFs uploaded: {st.session_state.pdf_count}")

    # Assuming tab_mapping is defined as:
    tab_mapping = {
        "qa": "Q & A",
        "category_search": "Category Search",
    }

    # Define tabs using full titles for display
    tab_titles = list(tab_mapping.values())

    # Create tabs in Streamlit
    tabs = st.tabs(tab_titles)

    # Use the active tab identifier to decide which content to display
    active_tab_identifier = st.session_state.get('active_tab', 'qa')  # Default to 'qa' if not set

    # Iterate over tabs and match with the active tab identifier for content display
    for idx, (identifier, title) in enumerate(tab_mapping.items()):
        with tabs[idx]:
            if identifier == "qa":
                st.subheader("Q & A")
                # Q & A specific content
                display_messages("qa")
                st.session_state["ingestion_spinner"] = st.empty()
                st.text_input("Ask questions relating to the paper",
                                key="user_input_qa",
                                on_change=process_input,
                                args=("qa",))
                
            elif identifier == "category_search":
                st.subheader("Category Search")
                # Category Search specific content
                display_messages("category_search")
                st.session_state["ingestion_spinner"] = st.empty()
                st.text_input("Enter a category or term within the paper to find relevant material & references",
                                key="user_input_category_search",
                                on_change=process_input,
                                args=("category_search",))

if __name__ == "__main__":
    page()