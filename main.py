import streamlit as st
from app.ui import pdf_uploader
from app.pdf_utils import extract_text_from_pdf
from app.vectorstore_utils import create_faiss_index, retrieve_relevant_docs
from app.chat_utils import get_chat_model, ask_chat_model
from app.config import API_KEY
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time
import os
from dotenv import load_dotenv
load_dotenv()

#load api key from environment variable
API_KEY = os.getenv("API_KEY")
st.set_page_config(
    page_title="Medical Document Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
    /* Blood Theme for Streamlit */
    
    /* Main container */
    .main .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Blood Theme Chat Messages */
    .chat-message {
        padding: 1.25rem;
        border-radius: 1rem;
        margin-bottom: 1.25rem;
        display: flex;
        flex-direction: row;
        align-items: flex-start;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(139, 0, 0, 0.3);
        background: linear-gradient(135deg, #2c0000 0%, #1a0000 100%);
        color: #ffcccc;
    }
    
    .chat-message::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 5px;
        height: 100%;
        background: linear-gradient(to bottom, #ff0000, #8b0000);
        box-shadow: 0 0 8px rgba(255, 0, 0, 0.5);
    }
    
    .chat-message.user {
        margin-left: 2rem;
        border-top-left-radius: 0;
    }
    
    .chat-message.assistant {
        margin-right: 2rem;
        border-top-right-radius: 0;
        background: linear-gradient(135deg, #1a0000 0%, #0d0000 100%);
        border: 1px solid rgba(85, 0, 0, 0.4);
    }
    
    .chat-message.assistant::before {
        background: linear-gradient(to bottom, #8b0000, #550000);
    }
    
    /* Blood Theme Avatars */
    .chat-message .avatar {
        width: 2.5rem;
        height: 2.5rem;
        border-radius: 50%;
        margin-right: 1rem;
        flex-shrink: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        box-shadow: 0 2px 8px rgba(0,0,0,0.4);
        border: 2px solid #8b0000;
        background: linear-gradient(135deg, #550000 0%, #330000 100%);
        color: #ff9999;
        font-size: 1.1rem;
    }
    
    .user .avatar {
        background: linear-gradient(135deg, #ff0000 0%, #8b0000 100%);
        color: white;
        box-shadow: 0 0 12px rgba(255, 0, 0, 0.5);
    }
    
    /* Message Content */
    .chat-message .message-content {
        flex: 1;
        padding-top: 0.25rem;
    }
    
    .chat-message .message {
        line-height: 1.6;
        font-size: 1rem;
        text-shadow: 0 0 2px rgba(0,0,0,0.7);
    }
    
    .chat-message .timestamp {
        font-size: 0.75rem;
        opacity: 0.8;
        margin-top: 0.75rem;
        text-align: right;
        font-weight: 500;
        color: #cc6666;
    }
    
    /* Blood Theme Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #ff0000 0%, #cc0000 100%);
        color: white;
        border-radius: 0.75rem;
        border: none;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 10px rgba(204, 0, 0, 0.4);
        transition: all 0.2s ease;
        border: 1px solid #8b0000;
        text-transform: uppercase;
        font-size: 0.9rem;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #ff3333 0%, #e60000 100%);
        box-shadow: 0 6px 15px rgba(204, 0, 0, 0.5);
        transform: translateY(-2px);
        border: 1px solid #ff4d4d;
    }
    
    .stButton > button:active {
        transform: translateY(0);
        box-shadow: 0 2px 5px rgba(204, 0, 0, 0.4);
    }
    
    /* Blood Theme Upload Section */
    .upload-section {
        background: linear-gradient(135deg, #2c0000 0%, #1a0000 100%);
        padding: 1.75rem;
        border-radius: 1rem;
        margin-bottom: 1.75rem;
        box-shadow: 0 6px 18px rgba(0,0,0,0.35);
        border: 1px solid #550000;
        color: #ffcccc;
    }
    
    .upload-section h3 {
        margin-top: 0;
        color: #ff6666;
        font-weight: 600;
        text-shadow: 0 0 3px rgba(255, 102, 102, 0.5);
        font-size: 1.4rem;
    }
    
    .upload-section p {
        color: #ff9999;
        line-height: 1.6;
    }
    
    /* Blood Theme Status Messages */
    .status-success {
        background: linear-gradient(135deg, #3d0000 0%, #2c0000 100%);
        color: #ff9999;
        padding: 1rem 1.25rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
        border-left: 4px solid #ff0000;
        box-shadow: 0 3px 10px rgba(0,0,0,0.25);
        font-weight: 500;
        border: 1px solid #8b0000;
    }
    
    /* Blood Theme Input Fields */
    input[class*="stTextInput"] {
        background: linear-gradient(135deg, #1a0000 0%, #0d0000 100%) !important;
        border: 1px solid #8b0000 !important;
        border-radius: 0.75rem !important;
        color: #ffcccc !important;
        box-shadow: 0 2px 8px rgba(139, 0, 0, 0.3) !important;
        padding: 0.75rem 1rem !important;
    }
    
    input[class*="stTextInput"]:focus {
        border-color: #ff0000 !important;
        box-shadow: 0 0 0 3px rgba(255, 0, 0, 0.3) !important;
    }
    
    /* Blood Theme Text Areas */
    textarea[class*="stTextArea"] {
        background: linear-gradient(135deg, #1a0000 0%, #0d0000 100%) !important;
        border: 1px solid #8b0000 !important;
        border-radius: 0.75rem !important;
        color: #ffcccc !important;
        box-shadow: 0 2px 8px rgba(139, 0, 0, 0.3) !important;
        padding: 1rem !important;
    }
    
    textarea[class*="stTextArea"]:focus {
        border-color: #ff0000 !important;
        box-shadow: 0 0 0 3px rgba(255, 0, 0, 0.3) !important;
    }
    
    /* Blood Theme Select Boxes */
    div[data-baseweb="select"] {
        background: linear-gradient(135deg, #1a0000 0%, #0d0000 100%) !important;
        border: 1px solid #8b0000 !important;
        border-radius: 0.75rem !important;
        box-shadow: 0 2px 8px rgba(139, 0, 0, 0.3) !important;
    }
    
    div[data-baseweb="select"]:hover {
        border-color: #cc0000 !important;
    }
    
    /* Blood Theme Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #ff6666 !important;
        text-shadow: 0 0 4px rgba(255, 102, 102, 0.4);
    }
    
    /* Blood Theme Links */
    a {
        color: #ff4d4d !important;
    }
    
    a:hover {
        color: #ff9999 !important;
        text-decoration: underline;
    }
    
    /* Blood Theme Scrollbars */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a0000;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #8b0000;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #ff0000;
    }
    
    /* Blood Drip Effect for User Messages */
    .chat-message.user::after {
        content: " ";
        position: absolute;
        top: 8px;
        right: -5px;
        width: 3px;
        height: 12px;
        background: #ff4d4d;
        border-radius: 0 0 50% 50%;
        opacity: 0.7;
        animation: drip 3s infinite;
    }
    
    @keyframes drip {
        0% { height: 12px; opacity: 0.7; }
        50% { height: 20px; opacity: 0.9; }
        100% { height: 12px; opacity: 0.7; }
    }
    
    /* Responsive Adjustments */
    @media (max-width: 768px) {
        .chat-message {
            padding: 1rem;
            margin-bottom: 1rem;
        }
        
        .chat-message.user {
            margin-left: 1rem;
        }
        
        .chat-message.assistant {
            margin-right: 1rem;
        }
        
        .upload-section {
            padding: 1.25rem;
        }
        
        .stButton > button {
            padding: 0.65rem 1.25rem;
            font-size: 0.85rem;
        }
    }
    
    /* Dark Mode Override */
    @media (prefers-color-scheme: dark) {
        body {
            background-color: #0d0000 !important;
        }
    }
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_model" not in st.session_state:
    st.session_state.chat_model = None
    
st.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <h1 style="color: #ff4b4b; font-size: 3rem; margin-bottom: 0.5rem;">🏥 MediChat Pro</h1>
    <p style="font-size: 1.2rem; color: #666; margin-bottom: 2rem;">Your Intelligent Medical Document Assistant</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for document upload
with st.sidebar:
    
    st.markdown("### 📁 Document Upload")
    st.markdown("Upload your medical documents to start chatting!")
    
    uploaded_files = pdf_uploader()
    
    if uploaded_files:
        st.success(f"📄 {len(uploaded_files)} document(s) uploaded")
        
        # Process documents
        if st.button("Process Documents", type="primary"):
            with st.spinner("Processing your medical documents..."):
                # Extract text from all PDFs
                all_texts = []
                for file in uploaded_files:
                    text = extract_text_from_pdf(file)
                    all_texts.append(text)
                
                # Split texts into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len,
                )
                
                chunks = []
                for text in all_texts:
                    chunks.extend(text_splitter.split_text(text))
                
                # Create FAISS index

                vectorstore = create_faiss_index(chunks)
                st.session_state.vectorstore = vectorstore
                
                # Initialize chat model
                chat_model = get_chat_model(API_KEY)
                st.session_state.chat_model = chat_model
                
                st.success("✅ Documents processed successfully!")
                st.balloons()

# Main chat interface
st.markdown("### 💬 Chat with Your Medical Documents")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        st.caption(message["timestamp"])

# Chat input
if prompt := st.chat_input("Ask about your medical documents..."):
    # Add user message to chat history
    timestamp = time.strftime("%H:%M")
    st.session_state.messages.append({
        "role": "user", 
        "content": prompt, 
        "timestamp": timestamp
    })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
        st.caption(timestamp)
    
    # Generate response
    if st.session_state.vectorstore and st.session_state.chat_model:
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching documents..."):
                # Retrieve relevant documents
                relevant_docs = retrieve_relevant_docs(st.session_state.vectorstore, prompt)

                # Create context from relevant documents
                context = "\n\n".join([doc.page_content for doc in relevant_docs])
                
                # Create prompt with context
                system_prompt = f"""You are MediChat Pro, an intelligent medical document assistant. 
                Based on the following medical documents, provide accurate and helpful answers. 
                If the information is not in the documents, clearly state that.
                When you are giving the answer, make sure that you try to take help of llm and give me a full diagnosis of the problem.
                Medical Documents:
                {context}

                User Question: {prompt}

                Answer:"""
                
                response = ask_chat_model(st.session_state.chat_model, system_prompt)
            
            st.markdown(response)
            st.caption(timestamp)
            
            # Add assistant message to chat history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response, 
                "timestamp": timestamp
            })
    else:
        with st.chat_message("assistant"):
            st.error("⚠️ Please upload and process documents first!")
            st.caption(timestamp)

# Footer
st.markdown("---")
