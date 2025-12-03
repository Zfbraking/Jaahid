from typing import Dict, Any
from tools.mcp_tooling import MCPTool

def build_summary_tool(llm) -> MCPTool:
    def _run(payload: Dict[str, Any]) -> Dict[str, Any]:
        text = payload.get("text", "")
        style = payload.get("style", "concise")
        if not text:
            return {"summary": ""}

        # IMPORTANT: ask the LLM to avoid Markdown / ** / special formatting
        prompt = (
            "Summarize the following content.\n"
            f"Write in a {style} style.\n"
            "Output PLAIN TEXT ONLY.\n"
            "Do NOT use Markdown, asterisks, bold (**), backticks, or any other special formatting.\n"
            "You may use numbered lists like '1.' and '-' bullets, but no markdown syntax.\n\n"
            "Content to summarize:\n"
            f"{text}"
        )

        resp = llm.invoke(prompt)
        summary = resp.content if hasattr(resp, "content") else str(resp)
        return {"summary": summary}

    return MCPTool(
        name="summarizer",
        description="Summarizes provided text in plain text (no Markdown).",
        func=_run
    )
