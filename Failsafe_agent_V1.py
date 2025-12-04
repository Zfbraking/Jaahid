import os, uuid, re, ast, unicodedata
from collections import Counter
from math import sqrt
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

ALLOWED_MODELS = {"gpt-4o-mini", "gpt-4o"}
MAX_OUTPUT_CHARS = 4000

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

# -----------------------------
# Helpers
# -----------------------------
def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).strip()

def _truncate(s: str, limit: int) -> tuple[str, bool]:
    return (s[:limit], len(s) > limit)

def _extract_code_block(query: str) -> str | None:
    start = query.find("```")
    if start == -1: return None
    end = query.find("```", start + 3)
    if end == -1: return None
    return query[start + 3:end].strip()

# -----------------------------
# Input Restriction
# -----------------------------
SENSITIVE_CATEGORIES = {
    "self_harm": [r"\bsuicide\b", r"\bkill myself\b", r"\bself harm\b"],
    "violence": [r"\bkill\b", r"\bmake a bomb\b", r"\battack\b"],
    "illegal": [r"\bhack\b", r"\bcredit card\b", r"\bwifi password\b", r"\bcrack\b"],
    "personal_data": [r"\bssn\b", r"\baadhaar\b", r"\bpan\b", r"\bupi\b", r"\bpassword\b"],
}

def input_restriction_check(text: str, max_len: int = 5000) -> tuple[str,str]:
    if not text or len(text.strip()) < 3:
        return "blocked", "Empty/too short query"
    if len(text) > max_len:
        return "blocked", f"Query too long ({len(text)} chars). Limit: {max_len}"
    lower = text.lower()
    for cat, patterns in SENSITIVE_CATEGORIES.items():
        for p in patterns:
            if re.search(p, lower):
                return "blocked", f"Sensitive content: {cat}"
    return "passed", "OK"

# -----------------------------
# Malicious Code Check
# -----------------------------
DANGEROUS_PATTERNS = [
    r"\bos\.system\s*\(",
    r"\bsubprocess\.(Popen|run|call)\s*\(",
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"\brm\s+-rf\b",
]

FORBIDDEN_CALL_NAMES = {"eval","exec","compile","os.system","subprocess.Popen","subprocess.call","subprocess.run"}

def contains_dangerous_strings(s: str) -> list[str]:
    s_lower = s.lower()
    return [pat for pat in DANGEROUS_PATTERNS if re.search(pat, s_lower)]

def malicious_code_check(query: str, code_snippet: str | None = None) -> tuple[str,str]:
    hits = contains_dangerous_strings(query)
    if hits:
        return "blocked", f"Dangerous patterns: {hits}"
    if code_snippet:
        try:
            tree = ast.parse(code_snippet)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALL_NAMES:
                        return "blocked", f"Forbidden call: {node.func.id}"
        except SyntaxError:
            return "blocked", "Syntax error in code block"
    return "passed", "OK"

# -----------------------------
# Plagiarism Check
# -----------------------------
def _normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", "", s)
    return s.strip()

def cosine_token(a: str, b: str) -> float:
    A = Counter(_normalize_text(a).split())
    B = Counter(_normalize_text(b).split())
    dot = sum(A[t]*B[t] for t in set(A)|set(B))
    magA = sqrt(sum(v*v for v in A.values()))
    magB = sqrt(sum(v*v for v in B.values()))
    return dot/(magA*magB+1e-9)

def plagiarism_check(generated: str, corpus: list[str], cosine_thresh: float=0.85) -> tuple[str,str]:
    best = {"source_idx":None,"cosine":0.0}
    for i,ref in enumerate(corpus):
        cos = cosine_token(generated,ref)
        if cos>best["cosine"]:
            best={"source_idx":i,"cosine":cos}
    if best["cosine"]>=cosine_thresh:
        return "flagged", f"Similarity {best['cosine']:.2f} with corpus[{best['source_idx']}]"
    return "passed", "OK"

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

# -----------------------------
# Dummy LLM + Safety Filter
# -----------------------------
class DummyLLM:
    def __init__(self,name,key): self.name=name; self.key=key
    def invoke(self,prompt): return {"content":f"Echo from {self.name}: {prompt[:200]}"}

def llm_safety_filter(llm, query: str) -> tuple[str,str]:
    if "hack" in query.lower() or "rm -rf" in query.lower():
        return "blocked", "LLM safety filter flagged"
    return "passed", "OK"

# -----------------------------
# Pipeline with Failsafe
# -----------------------------
def answer_pipeline(user_query: str, reference_corpus: list[str], model_name: str="gpt-4o-mini"):
    trace_id = str(uuid.uuid4())[:8]
    user_query = _normalize(user_query)
    results = {}

    # Input restriction
    status, reason = input_restriction_check(user_query)
    results["Input Restriction"] = {"status": status, "reason": reason}
    if status == "blocked":
        return {"trace": trace_id, "checks": results, "final": "blocked"}

    # Malicious code
    code_block = _extract_code_block(user_query)
    status, reason = malicious_code_check(user_query, code_block)
    results["Malicious Code"] = {"status": status, "reason": reason}
    if status == "blocked":
        return {"trace": trace_id, "checks": results, "final": "blocked"}

    # LLM safety filter
    llm = DummyLLM(model_name, "dummy_key")
    status, reason = llm_safety_filter(llm, user_query)
    results["LLM Safety Filter"] = {"status": status, "reason": reason}
    if status == "blocked":
        return {"trace": trace_id, "checks": results, "final": "blocked"}

    # Failsafe LLM invocation
    try:
        resp = llm.invoke(user_query)
        content, _ = _truncate(resp["content"], MAX_OUTPUT_CHARS)
        results["LLM Invocation"] = {"status": "ok", "reason": "Primary succeeded"}
    except Exception as e:
        # Switch to backup model
        backup_llm = DummyLLM("gpt-4o", "dummy_key")
        resp = backup_llm.invoke(user_query)
        content, _ = _truncate(resp["content"], MAX_OUTPUT_CHARS)
        results["LLM Invocation"] = {"status": "fallback", "reason": f"Primary failed: {type(e).__name__} - {e}"}

    # Plagiarism
    status, reason = plagiarism_check(content, reference_corpus)
    results["Plagiarism"] = {"status": status, "reason": reason}
    if status == "flagged":
        return {"trace": trace_id, "checks": results, "final": "plagiarism_flagged"}

    results["Answer"] = {"status": "ok", "content": content}
    return {"trace": trace_id, "checks": results, "final": "ok"}


# ============================================================
# Gradio Interface with Scenario Dropdown
# ============================================================

import gradio as gr

# Simulated LLM for demo purposes
class SimulatedLLM:
    def __init__(self, name, fail_mode=None):
        self.name = name
        self.fail_mode = fail_mode

    def invoke(self, query):
        if self.fail_mode == "invalid_model":
            raise ValueError("Invalid model name")
        elif self.fail_mode == "network":
            raise ConnectionError("Network unreachable")
        elif self.fail_mode == "auth":
            raise PermissionError("401 Unauthorized - bad API key")
        elif self.fail_mode == "rate_limit":
            raise Exception("429 Too Many Requests")
        elif self.fail_mode == "service_down":
            raise Exception("Service unavailable")
        elif self.fail_mode == "bad_input":
            raise Exception("Malformed input payload")
        elif self.fail_mode == "runtime":
            raise RuntimeError("Unexpected runtime error")
        else:
            return {"content": f"Echo from {self.name}: {query}"}

def safe_invoke_with_simulation(query, fail_mode):
    """Failsafe wrapper using simulated LLMs."""
    primary_llm = SimulatedLLM("Primary-LLM", fail_mode=fail_mode)
    try:
        resp = primary_llm.invoke(query)
        return {
            "status": "ok",
            "reason": "Primary succeeded",
            "content": resp["content"]
        }
    except Exception as e:
        backup_llm = SimulatedLLM("Backup-LLM")
        resp = backup_llm.invoke(query)
        return {
            "status": "fallback",
            "reason": f"Primary failed: {type(e).__name__} - {e}",
            "content": resp["content"]
        }

# Gradio UI
def run_agent(user_query, scenario):
    return safe_invoke_with_simulation(user_query, scenario)

demo = gr.Interface(
    fn=run_agent,
    inputs=[
        gr.Textbox(lines=3, placeholder="Enter your query here..."),
        gr.Dropdown(
            choices=[
                "none", "invalid_model", "network", "auth",
                "rate_limit", "service_down", "bad_input", "runtime"
            ],
            value="none",
            label="Failure Scenario"
        )
    ],
    outputs="json",
    title="🛡️ Failsafe Agent MCP",
    description="Test different failure scenarios. Select a scenario from the dropdown to simulate primary LLM failure and see backup logic kick in."
)

if __name__ == "__main__":
    # CLI demo
    test_queries = {
        "Safe Query": "Explain quantum computing basics",
        "Sensitive Query": "I want to kill myself",
        "Malicious Code": "import os\nos.system('rm -rf /')",
        "Plagiarism": "Echo from gpt-4o-mini: This is an example paper text ...",
        "Oversized Input": "A" * 6000
    }

    print("\n=== Failsafe Agent Demo (CLI) ===\n")
    for label, query in test_queries.items():
        result = answer_pipeline(query, REFERENCE_CORPUS)
        print(f"--- {label} ---")
        print(f"Trace ID: {result['trace']}")
        print("| Check Category       | Status   | Reason/Content |")
        print("|----------------------|----------|----------------|")
        for check, outcome in result["checks"].items():
            reason = outcome.get("reason","")
            if check=="Answer":
                reason = outcome.get("content","")
            print(f"| {check:<20} | {outcome['status']:<8} | {reason[:60]} |")
        print(f"Final Outcome: {result['final']}\n")

    # Launch GUI
    demo.launch()