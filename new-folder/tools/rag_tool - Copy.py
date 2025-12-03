from typing import Dict, Any
from langchain.chains import RetrievalQA
from tools.mcp_tooling import MCPTool

def build_rag_tool(llm, retriever) -> MCPTool:
    def _run(payload: Dict[str, Any]) -> Dict[str, Any]:
        query = payload.get("query", "")
        if not query:
            return {"result": "", "sources": []}

        # Build QA chain fresh each run to avoid stale retriever
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )
        result = qa_chain.invoke({"query": query})

        print("[DEBUG] QA chain result keys:", result.keys())
        if "source_documents" in result:
            for doc in result["source_documents"][:1]:
                print("[DEBUG] Source doc preview:", doc.page_content[:200])

        return {
            "result": result.get("result", ""),
            "sources": result.get("source_documents", [])
        }

    return MCPTool(
        name="pdf_retriever",
        description="Retrieve and answer questions from the indexed PDF.",
        func=_run
    )
