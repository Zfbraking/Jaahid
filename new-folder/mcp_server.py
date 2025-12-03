from flask import Flask, request, jsonify, render_template
import os, requests, traceback
from tools.mcp_tooling import MCPRegistry
from tools.rag_tool import build_rag_tool
from tools.summary_tool import build_summary_tool
from llm_setup import get_llm, get_embeddings
from vectorstore import build_vectorstore, get_retriever
from tools.excel_tool import build_excel_tool
# === ML EXTENSION: START (import) ===
from tools.ml_tool import build_ml_tool
# === ML EXTENSION: END (import) ===
from tools.req2tc_tool import build_req2tc_tool  # REQ→TC EXTENSION
from tools.risk_tool import build_risk_tool  # EVENT RISK EXTENSION




app = Flask(__name__)

# Initialize LLM + embeddings
try:
    llm = get_llm()
    print("[DEBUG] LLM initialized")
except Exception as e:
    print("[FATAL] Failed to initialize LLM:", e)
    traceback.print_exc()
    llm = None

try:
    embeddings = get_embeddings()
    print("[DEBUG] Embeddings initialized")
except Exception as e:
    print("[FATAL] Failed to initialize embeddings:", e)
    traceback.print_exc()
    embeddings = None

registry = MCPRegistry()

# Register static tools
if llm is not None:
    registry.register(build_summary_tool(llm))
else:
    print("[WARN] LLM is None, summarizer tool will not be available.")

registry.register(build_excel_tool())

# === ML EXTENSION: START (tool registration) ===
# Register the ML modeler tool (training + prediction on Excel/CSV)
registry.register(build_ml_tool())
# === ML EXTENSION: END (tool registration) ===

# === REQ→TC EXTENSION: TOOL REGISTRATION START ===
# Requirement-to-Testcase generator tool
registry.register(build_req2tc_tool(llm, embeddings))
# === REQ→TC EXTENSION: TOOL REGISTRATION END ===


# === EVENT RISK EXTENSION: TOOL REGISTRATION START ===
registry.register(build_risk_tool())
# === EVENT RISK EXTENSION: TOOL REGISTRATION END ===


@app.route("/")
def home():
    return "<h1>✅ MCP Server is Running</h1>"


@app.route("/register_pdf", methods=["POST"])
def register_pdf():
    """
    Build a fresh vectorstore + retriever for the uploaded PDF
    and (re)register the `pdf_retriever` tool to point to this data.
    Uses embeddings + Chroma.
    """
    try:
        if llm is None:
            return jsonify({"error": "LLM is not initialized on MCP server."}), 500
        if embeddings is None:
            return jsonify({"error": "Embeddings model is not initialized on MCP server."}), 500

        data = request.get_json(force=True, silent=True) or {}
        chunks = data.get("chunks", [])

        if not chunks:
            return jsonify({"error": "No chunks provided to register_pdf."}), 400

        print(f"[DEBUG] /register_pdf received {len(chunks)} chunks")

        # Build vectorstore (Chroma + embeddings)
        vectordb = build_vectorstore(chunks, embeddings)
        retriever = get_retriever(vectordb, k=5)

        # Overwrite pdf_retriever tool to always point to latest PDF
        registry.register(build_rag_tool(llm, retriever))

        return jsonify({"status": "pdf retriever registered", "chunks": len(chunks)})
    except Exception as e:
        print("[ERROR] Exception in /register_pdf:", e)
        traceback.print_exc()
        return jsonify({"error": f"register_pdf failed: {e}"}), 500


@app.route("/invoke/<tool_name>", methods=["POST"])
def invoke_tool(tool_name):
    payload = request.json or {}
    tool = registry.get(tool_name)
    if not tool:
        return jsonify({"error": f"Tool {tool_name} not found"}), 404
    try:
        result = tool.invoke(payload)
        if not isinstance(result, dict):
            result = {"error": "Tool did not return a dict"}
    except Exception as e:
        print(f"[ERROR] Tool {tool_name} failed:", e)
        traceback.print_exc()
        result = {"error": f"Tool failed: {e}"}
    return jsonify(result)


# Optional Excel visuals route (if hitting MCP directly)
MCP_SERVER = "http://127.0.0.1:8000"


@app.route("/generate_visuals", methods=["POST"])
def generate_visuals():
    file = request.files.get("excel_file")
    file_path = request.form.get("file_path")

    if file:
        os.makedirs("uploads", exist_ok=True)
        save_path = os.path.join("uploads", file.filename)
        file.save(save_path)
        file_path = save_path

    if not file_path:
        return render_template("visuals.html", error="Please upload an Excel file.")

    # Decide mode
    if request.form.get("prompt"):
        payload = {
            "file_path": file_path,
            "mode": "prompt",
            "prompt": request.form.get("prompt"),
        }
    elif not request.form.get("chart_type"):
        payload = {"file_path": file_path, "suggestions_only": True}
    else:
        payload = {
            "file_path": file_path,
            "mode": "generate",
            "chart_type": request.form.get("chart_type"),
            "x_column": request.form.get("x_column"),
            "y_column": request.form.get("y_column"),
        }

    try:
        res = requests.post(
            f"{MCP_SERVER}/invoke/excel_visualizer", json=payload, timeout=90
        )
        try:
            visuals = res.json()
        except ValueError:
            visuals = {
                "error": f"Non-JSON response ({res.status_code}): {res.text[:200]}"
            }
    except Exception as e:
        print("[ERROR] /generate_visuals backend call failed:", e)
        traceback.print_exc()
        visuals = {"error": f"Backend call failed: {e}"}

    return render_template("visuals.html", visuals=visuals, file_path=file_path)


if __name__ == "__main__":
    app.run(port=8000, debug=True, use_reloader=False)
