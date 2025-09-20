from euriai.langchain import create_chat_model
# from app.config import API_KEY
import os
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise ValueError("API_KEY environment variable not set")

def get_chat_model(api_key: str):
    return create_chat_model(model="gpt-4.1-nano", api_key=API_KEY,temperature=0.7)

def ask_chat_model(chat_model, prompt: str):
    response =  chat_model.invoke(prompt)
    return response.content