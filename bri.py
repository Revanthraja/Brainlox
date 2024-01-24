import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load data using Langchain
loader = WebBaseLoader("https://brainlox.com/courses/category/technical")
docs = loader.load()

# Prepare Langchain components
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)

prompt = PromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

document_chain = create_stuff_documents_chain(model, prompt)

retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Streamlit App
st.title("Langchain Chatbot")

# User Input
user_input = st.text_input("Ask a question:")
context = st.text_area("Context (optional):")

# Invoke the Langchain chatbot
if st.button("Get Answer"):
    input_data = {"input": user_input, "context": [Document(page_content=context)]}
    response = retrieval_chain.invoke(input_data)
    st.write("Answer:", response["answer"])
