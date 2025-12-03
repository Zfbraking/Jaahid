# ====================================
# === REQ→TC EXTENSION: TOOL START ===
# ====================================
"""
Requirement-to-Testcase generator.

Tool name: req2tc_generator

Modes:
  - "index": load old requirements + testcases as examples
             payload: { "mode": "index", "entries": [ { "requirement": str, "testcases": str }, ... ] }

  - "generate": given a new requirement, retrieve similar old ones and
                generate new test cases in plain text.
             payload: { "mode": "generate", "requirement": str }

To remove this feature:
  - Delete this file
  - Remove its registration from mcp_server.py
  - Remove /req2tc route and req2tc.html from the Flask app
"""

from typing import Dict, Any, List

from tools.mcp_tooling import MCPTool
from vectorstore import build_vectorstore, get_retriever


# Global retriever for requirement→testcase examples
_REQ_RETRIEVER = None


def build_req2tc_tool(llm, embeddings) -> MCPTool:
    """
    Build the Requirement→Testcase generator tool.

    llm: ChatOpenAI instance
    embeddings: OpenAIEmbeddings instance
    """

    def _run(payload: Dict[str, Any]) -> Dict[str, Any]:
        mode = (payload.get("mode") or "generate").lower()

        if mode == "index":
            return _index_entries(payload, embeddings)
        elif mode == "generate":
            return _generate_testcases(payload, llm)
        else:
            return {"error": "Invalid mode. Use 'index' or 'generate'."}

    return MCPTool(
        name="req2tc_generator",
        description="Generate test cases for new requirements using old requirement→testcase examples.",
        func=_run,
    )


def _index_entries(payload: Dict[str, Any], embeddings) -> Dict[str, Any]:
    """
    Build a vector index from historical examples.

    payload expects:
      {
        "entries": [
          { "requirement": "...", "testcases": "Field1: ...\\nField2: ...\\n...\\n\\nField1: ...", },
          ...
        ]
      }

    Each 'testcases' string may contain multiple testcases for the same requirement,
    already formatted line-by-line by the Flask app.
    """
    global _REQ_RETRIEVER

    entries: List[Dict[str, Any]] = payload.get("entries") or []
    if not entries:
        return {"error": "No entries provided for indexing."}

    texts = []
    for e in entries:
        req = (e.get("requirement") or "").strip()
        tcs = (e.get("testcases") or "").strip()
        if not req:
            continue

        doc_text = (
            "Requirement:\n"
            f"{req}\n\n"
            "Associated test cases (with fields as provided in the mapping file):\n"
            f"{tcs if tcs else 'N/A'}"
        )
        texts.append(doc_text)

    if not texts:
        return {"error": "No valid requirement entries to index."}

    vectordb = build_vectorstore(texts, embeddings)
    _REQ_RETRIEVER = get_retriever(vectordb, k=5)

    return {"status": "ok", "indexed_count": len(texts)}



def _generate_testcases(payload: Dict[str, Any], llm) -> Dict[str, Any]:
    """
    Generate test cases for a new requirement, based on retrieved examples.

    The examples contain testcases as plain text with lines like:
      FieldName: value
      OtherField: other value

    Different projects may use different field names (e.g. TC_ID, Module, Type,
    Title, Preconditions, Steps, Expected, Priority, ASIL, etc.).

    The LLM is instructed to REUSE exactly the same field names and structure it
    sees in the examples, and not to invent or drop fields.
    """
    global _REQ_RETRIEVER

    requirement = (payload.get("requirement") or "").strip()
    if not requirement:
        return {"error": "Requirement text is empty."}

    if _REQ_RETRIEVER is None:
        return {"error": "No requirement index loaded. Upload mapping file and index first."}

    try:
        # Modern retrievers: use .invoke() instead of get_relevant_documents()
        docs = _REQ_RETRIEVER.invoke(requirement)
    except Exception as e:
        return {"error": f"Retrieval failed: {e}"}

    if not docs:
        examples_text = "No examples available."
    else:
        examples_text = ""
        for i, doc in enumerate(docs, start=1):
            examples_text += f"Example {i}:\n{doc.page_content}\n\n"

    prompt = (
        "You are a senior QA engineer.\n"
        "Below are example requirements and their associated test cases from this project.\n"
        "Each testcase is represented as PLAIN TEXT with multiple lines in the form:\n"
        "  FieldName: value\n"
        "  AnotherField: another value\n"
        "Different projects may use different field names, for example:\n"
        "  TC_ID, ReqID, Module, Type, Title, Preconditions, Steps, Expected,\n"
        "  Priority, ASIL_Level, SafetyGoal, etc.\n\n"
        "IMPORTANT:\n"
        "- For THIS project, you MUST reuse exactly the same field names and overall structure\n"
        "  that you see in the associated test cases of the examples below.\n"
        "- Do NOT invent new field names.\n"
        "- Do NOT drop existing fields that are present in the examples' testcases.\n"
        "- If examples show fields like 'TC_ID', 'Module', 'Type', 'Title', 'Preconditions',\n"
        "  'Steps', 'Expected', etc., then each generated testcase should use the same\n"
        "  set of fields and similar ordering.\n"
        "- Keep everything as plain text. NO markdown, no bullets using *, no #, no ```.\n\n"
        f"{examples_text}"
        "Now generate high-quality test cases for the NEW requirement below.\n\n"
        "NEW REQUIREMENT:\n"
        f"{requirement}\n\n"
        "Return ONLY the test cases, in the SAME PLAIN-TEXT FIELD-BASED FORMAT as the examples.\n"
    )

    try:
        resp = llm.invoke(prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)
    except Exception as e:
        return {"error": f"LLM call failed: {e}"}

    return {"testcases": text}


# ==================================
# === REQ→TC EXTENSION: TOOL END ===
# ==================================
