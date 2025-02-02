import streamlit as st
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
import ssl

# Disable SSL verification (Temporary Fix)
ssl._create_default_https_context = ssl._create_unverified_context

# Function to load local CSS
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("CSS file not found. Skipping custom styles.")

# Load custom CSS
local_css("static/style.css")

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Check if API keys are loaded properly
if not PINECONE_API_KEY:
    st.error("Error: PINECONE_API_KEY is missing!")
    st.stop()
if not GROQ_API_KEY:
    st.error("Error: GROQ_API_KEY is missing!")
    st.stop()

# Set API keys in environment
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Initialize embeddings and document search
try:
    embeddings = download_hugging_face_embeddings()
    index_name = "healtcare-chatbot"
    docsearch = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
except Exception as e:
    st.error(f"Error initializing Pinecone: {e}")
    st.stop()

# Create retrieval chain for answering questions
groq_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
question_answer_chain = create_stuff_documents_chain(groq_llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Build the Streamlit UI
st.title("🩺 Healthcare Chatbot")
st.markdown("Ask me anything about healthcare!")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    st.markdown(message, unsafe_allow_html=True)

# Input field
user_input = st.text_input("Your Message:", key="user_input")

if user_input:
    # Append user message
    st.session_state.messages.append(f"<div class='msg_cotainer_send'><strong>You:</strong> {user_input}</div>")

    # Retrieve response
    try:
        response = rag_chain.invoke({"input": user_input})
        bot_response = response.get("answer", "I'm sorry, I couldn't find an answer to that.")
    except Exception as e:
        bot_response = f"Error: {str(e)}"

    # Append bot response
    st.session_state.messages.append(f"<div class='msg_cotainer'><strong>Bot:</strong> {bot_response}</div>")

    # Rerun UI to update messages
    st.experimental_rerun()
