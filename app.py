import streamlit as st
import openai
from langchain_openai import ChatOpenAI, OpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Conversational ChatBot with OPENAI"

# Initialize conversation history
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = [
        {"role": "system", "content": "You are a helpful AI assistant named KpBot. Please respond to user questions."}
    ]

from langchain_core.prompts.chat import SystemMessage, HumanMessage, AIMessage

def generate_response(conversation_history, api_key, engine, temperature, max_tokens):
    openai.api_key = api_key
    llm = ChatOpenAI(temperature=temperature, max_tokens=max_tokens, model=engine)
    output_parser = StrOutputParser()

    # Convert conversation history into the correct format
    formatted_messages = []
    for message in conversation_history:
        if message["role"] == "system":
            formatted_messages.append(SystemMessage(content=message["content"]))
        elif message["role"] == "user":
            formatted_messages.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            formatted_messages.append(AIMessage(content=message["content"]))

    # Create a prompt from formatted messages
    prompt = ChatPromptTemplate.from_messages(formatted_messages)
    chain = prompt | llm | output_parser
    response = chain.invoke({})
    return response

# Streamlit UI
st.set_page_config(page_title="Conversational Chatbot", page_icon="🤖", layout="wide")

# Add custom CSS
st.markdown("""
    <style>
        .chat-container {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .user-message {
            background-color: #e1f5fe;
            border-radius: 10px;
            padding: 10px;
            max-width: 70%;
            margin:5%;
            align-self: flex-start;
        }
        .assistant-message {
            background-color: #f1f8e9;
            border-radius: 10px;
            padding: 10px;
            max-width: 70%;
            margin:5%
            align-self: flex-end;
        }
        .message-container {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .stTextInput input {
            padding: 10px;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            padding: 10px;
            border-radius: 5px;
            border: none;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar settings
st.sidebar.title("Settings")
api_key = os.getenv("OPENAI_API_KEY")
engine = st.sidebar.selectbox("Select OpenAI model", ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "text-davinci-003"])
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.55)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=240)

st.title("Conversational Chatbot with OpenAI")

st.write("Ask me anything and let's have a conversation!")

# Input field
user_input = st.text_input("You:", key="user_input", placeholder="Type your message here...")

if user_input:
    # Append user input to conversation history
    st.session_state.conversation_history.append({"role": "user", "content": user_input})
    
    # Generate response
    response = generate_response(
        st.session_state.conversation_history, api_key, engine, temperature, max_tokens
    )
    
    # Append response to conversation history
    st.session_state.conversation_history.append({"role": "assistant", "content": response})
    
    # Display the conversation in a more aesthetic way
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for message in st.session_state.conversation_history:
        if message["role"] == "user":
            st.markdown(f"<div class='user-message'>You : {message['content']}</div>", unsafe_allow_html=True)
        elif message["role"] == "assistant":
            st.markdown(f"<div class='assistant-message'>KpBot : {message['content']}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.write("You haven't entered any question yet!")
