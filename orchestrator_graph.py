from typing import TypedDict, Optional, Any
from langgraph.graph import StateGraph, END
from tools.mcp_tooling import MCPRegistry
from tools.rag_tool import build_rag_tool
from tools.summary_tool import build_summary_tool
from tools.citation_tool import build_citation_tool
from tools.web_tool import build_web_tool
from vectorstore import get_retriever

# Explicit state schema so LangGraph preserves keys
class AgentState(TypedDict):
    query: str
    llm: Any
    vectordb: Any
    embeddings: Any
    tools: Optional[Any]
    retrieved: Optional[str]
    sources: Optional[list]
    needs_web: Optional[bool]
    web_context: Optional[str]
    summary: Optional[str]
    citations: Optional[list]

def register_tools(state: AgentState):
    print("[DEBUG] Running node: register_tools | State keys:", list(state.keys()))
    if "vectordb" not in state or state["vectordb"] is None:
        raise ValueError("No vectordb found in state. Did you build the vectorstore?")
    tools = MCPRegistry()
    retriever = get_retriever(state["vectordb"], k=5)
    tools.register(build_rag_tool(state["llm"], retriever))
    tools.register(build_summary_tool(state["llm"]))
    tools.register(build_citation_tool())
    tools.register(build_web_tool())
    state["tools"] = tools
    return state

def retrieval_node(state: AgentState):
    print("[DEBUG] Running node: retrieval")
    tool = state["tools"].get("pdf_retriever")
    out = tool.invoke({"query": state["query"]})
    state["retrieved"] = out.get("result", "")
    state["sources"] = out.get("sources", [])
    return state

def decision_node(state: AgentState):
    print("[DEBUG] Running node: decision")
    text_len = len(state.get("retrieved", "") or "")
    needs_web = text_len < 200 and len(state.get("query", "")) > 10
    state["needs_web"] = needs_web
    return state

def web_node(state: AgentState):
    print("[DEBUG] Running node: web")
    tool = state["tools"].get("web_search")
    out = tool.invoke({"query": state["query"]})
    state["web_context"] = out.get("web_context", "")
    return state

def summarizer_node(state: AgentState):
    print("[DEBUG] Running node: summarizer")
    tool = state["tools"].get("summarizer")
    base_text = state.get("retrieved", "")
    web_text = state.get("web_context", "")
    combined = base_text if not web_text else f"{base_text}\n\nExternal context:\n{web_text}"
    out = tool.invoke({"text": combined, "style": "concise and structured"})
    state["summary"] = out.get("summary", "")
    return state

def citation_node(state: AgentState):
    print("[DEBUG] Running node: citation")
    tool = state["tools"].get("citations")
    out = tool.invoke({"sources": state.get("sources", [])})
    state["citations"] = out.get("citations", [])
    return state

def build_graph():
    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("register_tools", register_tools)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("decision", decision_node)
    graph.add_node("web", web_node)
    graph.add_node("summarizer", summarizer_node)
    graph.add_node("citation", citation_node)

    # Entry and linear edges to avoid concurrent branches
    graph.set_entry_point("register_tools")
    graph.add_edge("register_tools", "retrieval")
    graph.add_edge("retrieval", "decision")

    # Conditional routing: either go to web or straight to summarizer
    def route_after_decision(state: AgentState):
        print("[DEBUG] Routing after decision")
        return "web" if state.get("needs_web", False) else "summarizer"

    graph.add_conditional_edges(
        "decision",
        route_after_decision,
        {
            "web": "web",
            "summarizer": "summarizer",
        },
    )

    # Web enriches → summarizer
    graph.add_edge("web", "summarizer")

    # Move citation AFTER summarizer to keep flow sequential
    graph.add_edge("summarizer", "citation")

    # End after citations
    graph.add_edge("citation", END)

    print("[DEBUG] Graph compiled with nodes:", graph.nodes)
    return graph.compile()
