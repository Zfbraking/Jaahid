
import os
import shutil
import tempfile
import streamlit as st
from pdfminer.high_level import extract_text
#from langchain_text_splitters import RecursiveCharacterTextSplitter  # FIXED IMPORT
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
import httpx
import tiktoken
import numpy as np
import httpx
from langchain_openai import ChatOpenAI

# ==============================
# CONFIG
# ==============================
BASE_URL = "https://genailab.tcs.in"
LLM_MODEL = "azure_ai/genailab-maas-DeepSeek-V3-0324"         # FIXED PREFIX
EMBED_MODEL = "azure/genailab-maas-text-embedding-3-large" # FIXED PREFIX
API_KEY = os.getenv("GENAILAB_API_KEY", "sk-QibkR-xrOxD2AsMgdaL0Pg")  # change when needed

tiktoken_cache_dir = "./token"
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir
client = httpx.Client(verify=False)
# ==============================
# FAILSAFE LOGIC
# ==============================
def get_llm(primary=True):
    """Return primary or backup LLM depending on flag."""
    if primary:
        return ChatOpenAI(
            base_url=BASE_URL,
            #model="azure/genailab-maas-gpt-4o",   # Primary model
            model="azure/genailab-maas-invalid-model",
            api_key=API_KEY,
            http_client=client,
            temperature=0.2,
        )
    else:
        return ChatOpenAI(
            base_url=BASE_URL,
            model="azure/genailab-maas-gpt-35-turbo",  # Backup model
            api_key=API_KEY,
            http_client=client,
            temperature=0.2,
        )

    
def safe_invoke(prompt, llm, query):
    """Try primary LLM, fallback to backup if it fails."""
    chain = prompt | llm
    try:
        return chain.invoke(query)
    except Exception as e:
        print(f"⚠️ Primary model failed: {e}")
        backup_llm = get_llm(primary=False)
        backup_chain = prompt | backup_llm
        return backup_chain.invoke(query)    

# ==============================
# DEMO USAGE
# ==============================
if __name__ == "__main__":
    # Build a simple chain (prompt → LLM)
    from langchain_core.prompts import ChatPromptTemplate

    llm = get_llm(primary=True)
    prompt = ChatPromptTemplate.from_template("Answer briefly: {question}")

    query = {"question": "Explain protein folding in one sentence."}
    answer = safe_invoke(prompt, llm, query)
    # Show only the model’s text
    print("✅ Final Answer:", answer.content)

    #st.write(answer.content)   # just the text

    print("✅ Final Answer:", answer)