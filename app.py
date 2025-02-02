import streamlit as st
from PIL import Image
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

# Set Streamlit page config (MUST BE THE FIRST STREAMLIT COMMAND)
st.set_page_config(page_title="Healthcare Chatbot", page_icon="🩺", layout="centered")

# Disable SSL verification (Temporary Fix)
ssl._create_default_https_context = ssl._create_unverified_context

# Function to load local CSS
def local_css(file_name):
    try:
        with open(file_name, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("CSS file not found. Skipping custom styles.")

# Load custom CSS for dark theme
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

# Load custom images
logo_image = Image.open("static/portrait-3d-female-doctor-photoaidcom-cropped.jpg")  # Path to your logo image
bot_avatar = Image.open("static/—Pngtree—beautiful lady doctor_14504911.png")  # Path to your custom bot image
user_avatar = Image.open("static/—Pngtree—user avatar placeholder white blue_6796231.png")  # Path to your custom user image

# Custom CSS for dark theme
st.markdown(
    """
    <style>
    .stApp {
        background-color: #1e1e1e;  /* Dark background */
        color: #ffffff;  /* White text */
    }

    .stTextInput > div > div > input {
        background-color: #2e2e2e;  /* Dark input field */
        color: #ffffff;  /* White text */
        border: 1px solid #444;  /* Dark border */
    }

    .stButton > button {
        background-color: #4CAF50;  /* Green button */
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
    }

    .stMarkdown h1 {
        color: #4CAF50;  /* Green title */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display logo and title
col1, col2 = st.columns([1, 4])
with col1:
    st.image(logo_image, width=110)  # Adjust logo width as needed
with col2:
    st.markdown("<h1 style='text-align: left;'>WELLMate ChatBot </h1>", unsafe_allow_html=True)

st.markdown("<p style='text-align:center;'>Your health matters—let’s heal together!</p>", unsafe_allow_html=True )


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to render custom styled messages
def render_message(message, role):
    if role == "user":
        st.markdown(f'<div style="background-color: #FFFFFF; padding: 10px; border-radius: 10px; margin: 5px 0; color: black;">{message}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="background-color: #FFFFFF; padding: 10px; border-radius: 10px; margin: 5px 0; color: black;">{message}</div>', unsafe_allow_html=True)

# Display chat history with custom avatars
for message in st.session_state.messages:
    render_message(message["content"], message["role"])

# Input field
if user_input := st.chat_input("Your Message:"):
    # Append user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display user message
    render_message(user_input, "user")

    # Retrieve bot response
    try:
        response = rag_chain.invoke({"input": user_input})
        bot_response = response.get("answer", "I'm sorry, I couldn't find an answer to that.")
    except Exception as e:
        bot_response = f"Error: {str(e)}"

    # Append bot response to chat history
    st.session_state.messages.append({"role": "assistant", "content": bot_response})

    # Display bot response
    render_message(bot_response, "assistant")
