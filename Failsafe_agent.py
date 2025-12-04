# ============================================================
# IMPORTS
# ============================================================
import os
import shutil
import tempfile
import streamlit as st
from pdfminer.high_level import extract_text
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
import httpx
import tiktoken
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================
BASE_URL = "https://genailab.tcs.in"
LLM_MODEL = "azure_ai/genailab-maas-DeepSeek-V3-0324"         # FIXED PREFIX
EMBED_MODEL = "azure/genailab-maas-text-embedding-3-large"    # FIXED PREFIX
API_KEY = os.getenv("GENAILAB_API_KEY", "sk-QibkR-xrOxD2AsMgdaL0Pg")  # change when needed

tiktoken_cache_dir = "./token"
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir
client = httpx.Client(verify=False)

# ============================================================
# VALIDATION LAYERS
# ============================================================
def validate_query(query: str):
    """
    Validate user query for malicious code or sensitive content.
    """
    restricted = ["password", "terrorism", "self-harm"]
    if any(word in query.lower() for word in restricted):
        raise PermissionError("❌ Sensitive query restriction triggered")
    if "eval(" in query or "os.system" in query:
        raise PermissionError("❌ Malicious code detected")
    return query

def validate_file(uploaded_file):
    """
    Validate uploaded file type.
    """
    if not uploaded_file.name.endswith(".pdf"):
        raise ValueError("❌ Unsupported file type. Please upload a PDF.")
    return extract_text(uploaded_file)

def check_plagiarism(text: str):
    """
    Dummy plagiarism check — replace with real similarity check if available.
    """
    # For demo, flag if text contains suspicious repeated phrase
    if "protein folding" in text.lower() and text.count("protein folding") > 1:
        return True
    return False

# ============================================================
# FAILSAFE LOGIC
# ============================================================
def get_llm(primary=True):
    """Return primary or backup LLM depending on flag."""
    if primary:
        return ChatOpenAI(
            base_url=BASE_URL,
            # model="azure/genailab-maas-gpt-4o",   # Example primary model
            model="azure/genailab-maas-invalid-model",  # Force failure for demo
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
    """
    Try primary LLM, fallback to backup if it fails.
    Includes reason logging.
    """
    chain = prompt | llm
    try:
        result = chain.invoke(query)
        # Plagiarism check after response
        if check_plagiarism(result.content):
            print("⚠️ Plagiarism detected, rephrasing with backup...")
            backup_llm = get_llm(primary=False)
            backup_chain = prompt | backup_llm
            result = backup_chain.invoke("Rephrase: " + query["question"])
        return result
    except Exception as e:
        print(f"⚠️ Primary model failed: {e}")
        backup_llm = get_llm(primary=False)
        backup_chain = prompt | backup_llm
        return backup_chain.invoke(query)

# ============================================================
# DEMO USAGE
# ============================================================
if __name__ == "__main__":
    # Build a simple chain (prompt → LLM)
    prompt = ChatPromptTemplate.from_template("Answer briefly: {question}")

    # Example query
    raw_query = "Explain protein folding in one sentence."
    try:
        validated_query = validate_query(raw_query)
        llm = get_llm(primary=True)
        answer = safe_invoke(prompt, llm, {"question": validated_query})

        # Show only the model’s text
        print("✅ Final Answer:", answer.content)

        # Optional: display in Streamlit if running as app
        # st.write(answer.content)

        # Print full object for debugging (includes metadata, tokens, etc.)
        print("✅ Final Answer (full object):", answer)

    except Exception as e:
        print(f"❌ Blocked or failed: {e}")