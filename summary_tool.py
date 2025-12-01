from typing import Dict, Any
from tools.mcp_tooling import MCPTool

def build_summary_tool(llm) -> MCPTool:
    def _run(payload: Dict[str, Any]) -> Dict[str, Any]:
        text = payload.get("text", "")
        style = payload.get("style", "concise")
        if not text:
            return {"summary": ""}
        prompt = f"Summarize the following in a {style} style:\n\n{text}"
        resp = llm.invoke(prompt)
        return {"summary": resp.content if hasattr(resp, "content") else str(resp)}
    return MCPTool(
        name="summarizer",
        description="Summarizes provided text.",
        func=_run
    )
