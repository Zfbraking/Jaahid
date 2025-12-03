from typing import TypedDict, Optional, Any, List, Dict
from langgraph.graph import StateGraph, END
import requests

# -------------------------------------------------
# Agent state shared between graph nodes
# -------------------------------------------------
class AgentState(TypedDict, total=False):
    # User input
    query: str

    # Orchestrator decision
    intent: str  # "rag" | "req2tc" | "risk"

    # Generic text to be summarized (used by RAG + risk)
    retrieved: Optional[str]

    # Final summarized answer (for RAG + risk)
    summary: Optional[str]

    # Raw outputs for specific tools
    req2tc_output: Optional[str]              # direct testcases (NO summarizer)
    risk_output: Optional[Dict[str, Any]]     # raw JSON from risk_forecaster

    # Optional PDF sources
    sources: Optional[List[Any]]


# MCP server that exposes: pdf_retriever, summarizer,
# req2tc_generator, risk_forecaster
MCP_SERVER = "http://127.0.0.1:8000"


# -------------------------------------------------
# Intent detection node
# -------------------------------------------------
def intent_node(state: AgentState) -> AgentState:
    """
    Decide which agent/tool to use based on the user's query.

    - "req2tc": if the user asks for test cases.
    - "risk":   if the user asks about chance / probability / next day/week/month.
    - "rag":    default → PDF Q&A / summarization.
    """
    q = state.get("query", "") or ""
    ql = q.lower()

    intent = "rag"  # default

    # Req → Testcases intent
    if any(w in ql for w in ["test case", "testcases", "testcase", "tc "]):
        intent = "req2tc"
    # Risk forecasting intent
    elif any(w in ql for w in ["chance", "probability", "probable", "forecast", "next week", "next day",
                               "tomorrow", "next month", "7 days", "30 days"]):
        intent = "risk"

    state["intent"] = intent
    print(f"[DEBUG] Intent detected: {intent} for query: {q!r}")
    return state


# -------------------------------------------------
# PDF RAG retrieval node
# -------------------------------------------------
def retrieval_node(state: AgentState) -> AgentState:
    """
    Call pdf_retriever to answer questions based on the indexed PDF.
    """
    print("[DEBUG] Running node: retrieval (pdf_retriever)")
    resp = requests.post(
        f"{MCP_SERVER}/invoke/pdf_retriever",
        json={"query": state["query"]},
        timeout=90,
    )
    out = resp.json()

    state["retrieved"] = out.get("result", "")
    state["sources"] = out.get("sources", [])

    return state


# -------------------------------------------------
# Summarizer node (used by RAG + risk)
# -------------------------------------------------
def summarizer_node(state: AgentState) -> AgentState:
    """
    Summarize whatever is in state["retrieved"] into a concise answer.

    Used for:
      - PDF RAG answers
      - Risk forecast explanation text
    """
    print("[DEBUG] Running node: summarizer")

    base_text = state.get("retrieved", "") or ""
    if not base_text:
        state["summary"] = ""
        return state

    try:
        resp = requests.post(
            f"{MCP_SERVER}/invoke/summarizer",
            json={
                "text": base_text,
                "style": "concise and structured",
            },
            timeout=90,
        )
        out = resp.json()
        state["summary"] = out.get("summary", "")
    except Exception as e:
        err = f"[ERROR] Summarizer failed: {e}"
        print(err)
        # Fallback: return raw text
        state["summary"] = base_text

    return state


# -------------------------------------------------
# Req → Testcase node (NO summarizer)
# -------------------------------------------------
def req2tc_node(state: AgentState) -> AgentState:
    """
    Call req2tc_generator to generate test cases for the given requirement.

    IMPORTANT:
      - We do NOT pass this through the summarizer.
      - The format is exactly what the training file taught the tool.
    """
    print("[DEBUG] Running node: req2tc (req2tc_generator)")

    requirement_text = state.get("query", "") or ""
    try:
        resp = requests.post(
            f"{MCP_SERVER}/invoke/req2tc_generator",
            json={
                "mode": "generate",
                "requirement": requirement_text,
            },
            timeout=120,
        )
        out = resp.json()
    except Exception as e:
        err_msg = f"Req→TC tool call failed: {e}"
        print("[ERROR]", err_msg)
        state["req2tc_output"] = err_msg
        return state

    if "error" in out:
        err_msg = f"Req→TC error: {out['error']}"
        print("[ERROR]", err_msg)
        state["req2tc_output"] = err_msg
    else:
        state["req2tc_output"] = out.get("testcases", "")

    return state


# -------------------------------------------------
# Risk forecasting node (risk_forecaster) + summarizer
# -------------------------------------------------
def risk_node(state: AgentState) -> AgentState:
    """
    Call risk_forecaster to estimate expected count and probability of events.

    For now we use a simple heuristic:
      - Dimension: "severity" (CRITICAL/ERROR/WARNING/INFO) inferred from text.
      - Window:
          "tomorrow"/"next day" -> day
          "next week"/"7 days"  -> week
          "next month"/"30 days"-> month
          default -> week

    You can later extend this to parse service/error_code from the query.
    """
    print("[DEBUG] Running node: risk (risk_forecaster)")
    q = state.get("query", "") or ""
    ql = q.lower()

    # Dimension & value (severity-based heuristic)
    dimension = "severity"
    value = "CRITICAL"
    if "critical" in ql:
        value = "CRITICAL"
    elif "error" in ql:
        value = "ERROR"
    elif "warning" in ql:
        value = "WARNING"
    elif "info" in ql:
        value = "INFO"

    # Window heuristic
    if any(w in ql for w in ["tomorrow", "next day"]):
        window = "day"
    elif any(w in ql for w in ["next month", "30 days"]):
        window = "month"
    elif any(w in ql for w in ["next week", "7 days"]):
        window = "week"
    else:
        window = "week"  # default

    payload = {
        "mode": "forecast",
        "dimension": dimension,
        "value": value,
        "window": window,
    }
    print(f"[DEBUG] risk_forecaster payload: {payload}")

    try:
        resp = requests.post(
            f"{MCP_SERVER}/invoke/risk_forecaster",
            json=payload,
            timeout=90,
        )
        out = resp.json()
    except Exception as e:
        err_msg = f"Risk tool call failed: {e}"
        print("[ERROR]", err_msg)
        state["retrieved"] = err_msg
        state["risk_output"] = None
        return state

    if "error" in out:
        err_msg = f"Risk tool error: {out['error']}"
        print("[ERROR]", err_msg)
        state["retrieved"] = err_msg
        state["risk_output"] = out
        return state

    # Build human-readable text that summarizer can polish
    expected = out.get("expected_count", 0.0)
    prob = out.get("probability_at_least_one", 0.0)
    total_events = out.get("total_events_for_value", 0)
    total_days = out.get("total_days_observed", 0)

    window_human = {
        "day": "tomorrow (next 1 day)",
        "week": "the next 7 days",
        "month": "the next 30 days",
    }.get(window, window)

    risk_text = (
        f"Risk forecast for {dimension} = {value} over {window_human}:\n"
        f"- Expected number of events: {expected:.2f}\n"
        f"- Probability of at least one event: {prob * 100:.1f}%\n"
        f"- Based on {total_events} historical events for this value "
        f"across {total_days} observed days.\n"
    )

    state["retrieved"] = risk_text
    state["risk_output"] = out
    return state


# -------------------------------------------------
# Build the LangGraph
# -------------------------------------------------
def build_graph():
    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("intent", intent_node)
    graph.add_node("retrieval", retrieval_node)     # PDF RAG
    graph.add_node("summarizer", summarizer_node)   # Shared summarizer
    graph.add_node("req2tc", req2tc_node)           # Requirement → Testcases (no summarizer)
    graph.add_node("risk", risk_node)               # Risk forecast → summarizer

    # Entry point
    graph.set_entry_point("intent")

    # Conditional routing based on detected intent
    def route_after_intent(state: AgentState) -> str:
        intent = state.get("intent", "rag")
        if intent == "req2tc":
            return "req2tc"
        if intent == "risk":
            return "risk"
        return "retrieval"  # default: RAG

    graph.add_conditional_edges(
        "intent",
        route_after_intent,
        {
            "retrieval": "retrieval",
            "req2tc": "req2tc",
            "risk": "risk",
        },
    )

    # Flows:
    #   RAG:   intent → retrieval → summarizer → END
    #   RISK:  intent → risk       → summarizer → END
    #   ReqTC: intent → req2tc     → END  (NO summarizer)
    graph.add_edge("retrieval", "summarizer")
    graph.add_edge("risk", "summarizer")
    graph.add_edge("summarizer", END)
    graph.add_edge("req2tc", END)

    print("[DEBUG] Graph compiled with nodes:", graph.nodes)
    return graph.compile()
