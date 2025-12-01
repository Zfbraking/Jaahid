from flask import Flask, render_template, request, jsonify
import tempfile
import os
from llm_setup import get_llm, get_embeddings
from orchestrator_graph import build_graph, AgentState
from pdf_utils import extract_pdf_text, chunk_text
from vectorstore import build_vectorstore
import markdown

app = Flask(__name__)

# Initialize once
llm = get_llm()
embeddings = get_embeddings()

@app.route("/")
def index():
    return render_template("dashboard.html")

@app.route("/summarize", methods=["POST"])
def summarize():
    pdf = request.files.get("pdf")
    query = request.form.get("query", "").strip()

    if not pdf or not query:
        return jsonify({"error": "PDF and query are required"}), 400

    # Save uploaded file to a temporary path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf.save(tmp.name)
        temp_path = tmp.name

    try:
        # Extract text and build vectorstore fresh each request
        raw_text = extract_pdf_text(temp_path)
        chunks = chunk_text(raw_text, chunk_size=1000, overlap=200)
        vectordb = build_vectorstore(chunks, embeddings)
        docs = vectordb.similarity_search(query, k=1)
        print("Top doc preview:", docs[0].page_content[:200])


        # Build a fresh graph each time
        app_graph = build_graph()

        state = AgentState(
            query=query,
            llm=llm,
            vectordb=vectordb,
            embeddings=embeddings
        )
        result = app_graph.invoke(state)

        summary = result.get("summary", "")
        citations = result.get("citations", [])

        # Convert Markdown summary to HTML
        summary_html = markdown.markdown(summary)

        return jsonify({"summary": summary_html, "citations": citations})
    finally:
        os.remove(temp_path)

if __name__ == "__main__":
    app.run(debug=True)
