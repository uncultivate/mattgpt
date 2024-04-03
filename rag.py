# Import necessary libraries
import os
import tempfile
import logging
import shutil
import streamlit as st
from streamlit_chat import message
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever

GROQ_API_KEY= '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'

#chat = ChatGroq(temperature=0, groq_api_key="YOUR_API_KEY", model_name="mixtral-8x7b-32768")


API_KEY = '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'

# Configure basic logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Define the ChatPDF class

class ChatPDF:
    vector_store = None
    retriever = None
    chain1 = None
    chain2 = None

    def __init__(self):
        # self.model = ChatOllama(model="mistral")
        self.model = ChatGroq(temperature=0.5, groq_api_key="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", model_name="mixtral-8x7b-32768")

        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.prompts = {
            "qa": PromptTemplate.from_template(
                """
                <s> [INST] You are an assistant for question-answering tasks, specializing in contract review. Use the following pieces of retrieved context
                to answer the question. If you don't know the answer, say so. Use three sentences maximum and keep the answer concise. [/INST] </s>
                [INST] Question: {question}
                Context: {context}
                Answer: [/INST]
                """
            ),
            "category_search": PromptTemplate.from_template(
                # """
                # <s> [INST] Review the provided contract document thoroughly to identify and extract all terms, clauses, and appendices related to the specified category: {question}. Summarize these sections accurately, including direct quotations and precise references (section numbers, titles, page numbers) where applicable, using the following format:
                # * Summary
                # * References to relevant clauses and 1-2 sentence description of each
                # * Supplementary information 
                # [/INST] </s>
                # Context: {context}
                # Answer: [/INST]
                # """
                """
                Upon reviewing the provided contract document, concentrate on identifying and extracting all terms, clauses, and appendices related to the specified category: {question}. Your output should follow a structured and detailed format. Adhere to the instructions below for content organization and formatting:

                Summary of {question} Category
                Begin with a Summary section focused on the {question} category. Provide an accurate summary of the key points and objectives covered under this category, including the significance of these elements to the overall contract.
                Detailed Analysis
                Proceed with a Detailed Analysis section where you will:
                List and Describe Relevant Terms and Clauses: Identify each term and clause related to the {question} category. Use bullet points for each term or clause, including direct quotations and precise references such as section numbers, titles, and page numbers where applicable.

                Example:

                Clause 4.3 - Termination Rights (Page 7): "Either party may terminate this agreement with a 30-day written notice if..."
                Appendices Related to {question}: Identify any appendices that are directly relevant to the {question} category. Provide titles and page numbers, along with a brief description of the content and its relevance.

                Direct Quotations and References
                Throughout the analysis, incorporate Direct Quotations from the contract to support your summaries and descriptions. Ensure these quotations are precise and include exact references to their location in the document (e.g., "As stated in Section 2.1 (Page 3), '...'" ).
                Formatting and Organization
                Use headers and bullet points to organize the content clearly.
                Ensure the entire response is well-structured and easy to navigate, providing a comprehensive overview and detailed breakdown of the {question} category within the contract.
                This refined approach will enable a thorough and focused analysis of the specified category, aiding in the understanding of its terms, implications, and relevance to the contract as a whole.
                Context: {context}
                """
            ),
        }
        logging.info('ChatPDF class created.')


    def ingest(self, pdf_file_path: str):
        # Load PDF documents
        try:
            docs = PyPDFLoader(file_path=pdf_file_path).load()
        except Exception as e:
            logging.error(f"Failed to load PDF document: {e}")
            return

        # Split documents into manageable chunks
        try:
            chunks = self.text_splitter.split_documents(docs)
            chunks = filter_complex_metadata(chunks)
        except Exception as e:
            logging.error(f"Error during document splitting: {e}")
            return

        # Initialize or update the vector store with new documents
        if self.vector_store is None:
            try:
                self.vector_store = Chroma.from_documents(documents=chunks, embedding=FastEmbedEmbeddings())
            except Exception as e:
                logging.error(f"Failed to initialize vector store: {e}")
                return
        else:
            try:
                self.vector_store.add_documents(documents=chunks)
            except Exception as e:
                logging.error(f"Failed to add documents to vector store: {e}")
                return

        # Setup retriever and processing chains if not already done
        if self.retriever is None:
            try:
                self.retriever = self.vector_store.as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs={"k": 3, "score_threshold": 0.5},
                )

                # Since the prompt templates are now dynamic, ensure chains are set up correctly here
                self.setup_chains()
            except Exception as e:
                logging.error(f"Failed to setup retriever or processing chains: {e}")

        


    def setup_chains(self):
        # This method assumes that prompt templates and the model are already defined
        self.chain = {
            "qa": ({"context": self.retriever, "question": RunnablePassthrough()}
                   | self.prompts["qa"]
                   | self.model
                   | StrOutputParser()),
            "category_search": ({"context": self.retriever, "question": RunnablePassthrough()}
                                | self.prompts["category_search"]
                                | self.model
                                | StrOutputParser()),
        }
        logging.info('Processing chains set up successfully.')

    

    def ask(self, query: str, query_type: str):
        if not self.vector_store or not self.retriever:
            return "Please add a PDF document first."
        if query_type not in self.prompts:
            return "Invalid query type."

        logging.info(f'Processing {query_type} query: {query}')

        prompt = self.prompts[query_type]

        chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                 | prompt
                 | self.model
                 | StrOutputParser())

        return chain.invoke(query)

    def clear(self):
        # Reset the vector store, retriever, and processing chains
        self.vector_store = None
        self.retriever = None
        self.chain = None
        logging.info('ChatPDF instance cleared and ready for new documents.')