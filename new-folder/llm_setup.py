import os
import httpx
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load environment variables
load_dotenv()

# Configure tiktoken cache directory
tiktoken_cache_dir = "./token"
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir

# Always disable SSL verification
client = httpx.Client(verify=False)

def get_llm():
    return ChatOpenAI(
        base_url=os.getenv("LLM_BASE_URL"),
        model=os.getenv("LLM_MODEL"),
        api_key=os.getenv("LLM_API_KEY"),
        http_client=client
    )

def get_embeddings():
    return OpenAIEmbeddings(
        base_url=os.getenv("EMBED_BASE_URL"),
        model=os.getenv("EMBED_MODEL"),
        api_key=os.getenv("LLM_API_KEY"),
        http_client=client
    )

# Define llm function here, no self-import
def llm(prompt_text: str) -> str:
    chat = get_llm()
    response = chat.invoke(prompt_text)
    return response.content   # should be JSON string
