import os
import streamlit as st

from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_groq import ChatGroq
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_classic.prompts import PromptTemplate


load_dotenv()

st.set_page_config(page_title="RAG Chatbot")
st.title("PDF RAG Chatbot")


def read_pdf(pdf_files):
    text = ""

    for pdf in pdf_files:
        reader = PdfReader(pdf)

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

    return text


def create_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(text)
    return chunks


def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_texts(
        chunks,
        embedding=embeddings
    )

    vector_store.save_local("faiss_index")


def get_chain():
    prompt_template = """
    Answer the question using only the given context.
    If the answer is not available in the context, say:
    "I could not find the answer in the uploaded PDF."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    model = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    chain = load_qa_chain(
        model,
        chain_type="stuff",
        prompt=prompt
    )

    return chain


uploaded_files = st.file_uploader(
    "Upload your PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if st.button("Process PDFs"):
    if uploaded_files:
        with st.spinner("Reading and processing PDFs..."):
            raw_text = read_pdf(uploaded_files)
            chunks = create_chunks(raw_text)
            create_vector_store(chunks)

        st.success("PDFs processed successfully!")
    else:
        st.warning("Please upload at least one PDF.")


question = st.text_input("Ask a question from your PDFs")

if question:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = vector_store.similarity_search(question)

    chain = get_chain()

    response = chain(
        {
            "input_documents": docs,
            "question": question
        },
        return_only_outputs=True
    )

    st.write("### Answer")
    st.write(response["output_text"])