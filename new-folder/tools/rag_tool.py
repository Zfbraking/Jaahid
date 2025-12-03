from typing import Dict, Any, List
from langchain_classic.chains import RetrievalQA
from tools.mcp_tooling import MCPTool


def build_rag_tool(llm, retriever) -> MCPTool:
    """
    RAG tool using RetrievalQA over a retriever (Chroma + embeddings).
    Converts LangChain Document objects into plain JSON-serializable dicts.
    """

    def _run(payload: Dict[str, Any]) -> Dict[str, Any]:
        query = (payload.get("query") or "").strip()
        if not query:
            return {"result": "", "sources": []}

        # Build QA chain fresh each run to avoid stale retriever
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
        )
        result = qa_chain.invoke({"query": query})

        print("[DEBUG] QA chain result keys:", result.keys())

        raw_sources: List[Any] = result.get("source_documents", []) or []

        # Convert Document objects -> plain dicts
        sources: List[Dict[str, Any]] = []
        for doc in raw_sources:
            try:
                # LangChain Document typically has .page_content and .metadata
                content = getattr(doc, "page_content", str(doc))
                metadata = getattr(doc, "metadata", {}) or {}
                sources.append(
                    {
                        "content": content,
                        "metadata": metadata,
                    }
                )
            except Exception as e:
                print("[WARN] Failed to serialize source document:", e)
                sources.append({"content": str(doc), "metadata": {}})

        if raw_sources:
            print("[DEBUG] Source doc preview:", raw_sources[0].page_content[:200])

        answer = result.get("result", "")
        # Some chain configs use 'output_text'
        if not answer and "output_text" in result:
            answer = result["output_text"]

        return {
            "result": answer,
            "sources": sources,  # now JSON-serializable
        }

    return MCPTool(
        name="pdf_retriever",
        description="Retrieve and answer questions from the indexed PDF.",
        func=_run,
    )
