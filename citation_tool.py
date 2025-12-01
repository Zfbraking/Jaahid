from typing import Dict, Any
from tools.mcp_tooling import MCPTool

def build_citation_tool() -> MCPTool:
    def _run(payload: Dict[str, Any]) -> Dict[str, Any]:
        docs = payload.get("sources", [])
        # Produce succinct source previews; adapt to your doc schema
        cites = []
        for i, d in enumerate(docs):
            content = getattr(d, "page_content", "")
            meta = getattr(d, "metadata", {})
            preview = content[:300].replace("\n", " ").strip()
            cites.append({
                "id": i + 1,
                "preview": preview,
                "metadata": meta
            })
        return {"citations": cites}
    return MCPTool(
        name="citations",
        description="Extracts citation previews from source documents.",
        func=_run
    )
