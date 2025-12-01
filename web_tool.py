from typing import Dict, Any
from tools.mcp_tooling import MCPTool

def build_web_tool() -> MCPTool:
    def _run(payload: Dict[str, Any]) -> Dict[str, Any]:
        query = payload.get("query", "")
        # Placeholders — integrate proper search API if needed
        return {"web_context": f"(web search placeholder for: {query})"}
    return MCPTool(
        name="web_search",
        description="Fetches external context for queries (placeholder).",
        func=_run
    )
