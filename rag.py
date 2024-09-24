import os
import logging
from typing import List, Optional
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores.utils import filter_complex_metadata

# Attempt to use pysqlite3, fall back to sqlite3 if not available
import sys
try:
    import pysqlite3
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    import sqlite3

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

class ChatPDF:
    def __init__(self):
        self.model = ChatGroq(temperature=0.5, 
                              groq_api_key=st.secrets["GROQ_API_KEY"], 
                              model_name="mixtral-8x7b-32768")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.vector_store: Optional[Chroma] = None
        self.retriever = None
        self.chain = None
        self.setup_prompts()

    def setup_prompts(self):
        self.prompts = {
            "qa": PromptTemplate.from_template(
                """
                <s> [INST] You are an assistant for medical research tasks, specializing in literature review. Use the following pieces of retrieved context
                to answer the question. If you don't know the answer, just say so. Use three sentences maximum and keep the answer concise. [/INST] </s>
                [INST] Question: {question}
                Context: {context}
                Answer: [/INST]
                """
            ),
            "category_search": PromptTemplate.from_template(
                """
                <s> [INST] You are an advanced AI system designed to conduct comprehensive literature reviews in the medical field. Your primary function is to analyze medical literature, extract key information, and provide detailed summaries within specified categories. You should also identify and highlight any references relevant to the given category.[/INST]</s>
                [INST] Context: {context}[/INST]
                [INST]
                If there is no provided context, just say so and do not provide any further information or summary material, just end the response.
                If there is provided context, then concentrate on identifying and extracting all references related to the specified category: {question}. Your output should follow a structured and detailed format. Adhere to the instructions below for content organization and formatting: [/INST] 

                [INST]Key Responsibilities
                1. Analyze the given medical literature which may include peer-reviewed journals, clinical trials, systematic reviews, and meta-analyses.
                2. Categorize information into key areas as specified by the user (e.g., methodology, results, conclusions, implications).
                3. Provide clear and concise summaries for each category.
                4. Include direct quotations from the source material, ensuring proper citation.
                5. Identify and highlight references to the given category throughout the literature.
                6. Offer a brief analysis of the findings, including potential implications and areas for further research.

                Output Format:
                For each category, provide:

                1. A concise summary (100-200 words)
                2. 2-3 relevant direct quotations with proper citations
                3. A list of additional references to the category found in the literature
                4. A brief analysis (50-100 words) of the findings and their significance

                Guidelines:
                1. Maintain objectivity in your summaries and analysis.
                2. Use clear, professional language appropriate for a medical audience.
                3. Prioritize recent and high-impact studies when applicable.
                4. Highlight any conflicting findings or controversies in the literature.
                5. Identify gaps in current research or areas requiring further investigation.
                6. Provide proper citations for all information, following a standard medical citation format (e.g., AMA, Vancouver).[/INST]
                """
            ),
        }

    def ingest(self, pdf_file_path: str, k: int):
        try:
            docs = PyPDFLoader(file_path=pdf_file_path).load()
            chunks = self.text_splitter.split_documents(docs)
            chunks = filter_complex_metadata(chunks)

            if self.vector_store is None:
                self.vector_store = Chroma.from_documents(documents=chunks, embedding=FastEmbedEmbeddings())
            else:
                self.vector_store.add_documents(documents=chunks)

            self.setup_retriever(k)
            self.setup_chains()
            logging.info(f'Document ingested successfully: {pdf_file_path}')
        except Exception as e:
            logging.error(f"Error ingesting document: {e}")
            raise

    def setup_retriever(self, k: int):
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": k, "score_threshold": 0.5},
        )

    def setup_chains(self):
        self.chain = {
            query_type: (
                {"context": self.retriever, "question": RunnablePassthrough()}
                | self.prompts[query_type]
                | self.model
                | StrOutputParser()
            )
            for query_type in self.prompts
        }
        logging.info('Processing chains set up successfully.')

    def ask(self, query: str, query_type: str) -> str:
        if not self.vector_store or not self.retriever:
            raise ValueError("Please add a PDF document first.")
        if query_type not in self.prompts:
            raise ValueError("Invalid query type.")

        retrieved_context = self.retriever.get_relevant_documents(query)
        if not retrieved_context:
            raise ValueError(f'No relevant context found for {query_type} query: {query}')

        return self.chain[query_type].invoke(query)

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None
        logging.info('ChatPDF instance cleared and ready for new documents.')