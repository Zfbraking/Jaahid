import os
import requests
import pandas as pd
import json
from requests.exceptions import ReadTimeout, RequestException
from flask import Flask, render_template, request, redirect, url_for, session, jsonify

from pdf_utils import extract_pdf_text, chunk_text
from orchestrator_graph import build_graph

app = Flask(__name__)
app.secret_key = "supersecret"
MCP_SERVER = "http://127.0.0.1:8000"

# Compile LangGraph once for PDFs
PDF_GRAPH = build_graph()


# ---------- Core helpers ----------

def run_pdf_rag(file_path: str, query: str, is_new_upload: bool = False):
    """
    PDF path + query -> register embeddings (if new) -> LangGraph RAG -> summary.
    """
    if not query or not query.strip():
        return None, "Query cannot be empty."

    if is_new_upload:
        # Extract & chunk the PDF only when it's newly uploaded
        raw_text = extract_pdf_text(file_path)
        if not raw_text:
            return None, "Could not read the PDF or it is empty."

        chunks = chunk_text(raw_text)

        if not chunks:
            return None, "Could not split PDF into chunks."

        # Optional: cap chunks so it doesn't explode
        MAX_CHUNKS = 200
        if len(chunks) > MAX_CHUNKS:
            print(f"[WARN] PDF produced {len(chunks)} chunks, capping to {MAX_CHUNKS} for indexing.")
            chunks = chunks[:MAX_CHUNKS]

        # Register PDF in MCP server: pdf_retriever now points to this PDF
        try:
            reg_res = requests.post(
                f"{MCP_SERVER}/register_pdf",
                json={"chunks": chunks},
                timeout=300,
            )
            reg_res.raise_for_status()
        except ReadTimeout:
            return None, (
                "Indexing the PDF took too long and timed out. "
                "Try a smaller PDF, or split the document into parts."
            )
        except RequestException as e:
            return None, f"Failed to register PDF with backend: {e}"

    # Run LangGraph orchestrator: retrieval -> summarizer
    try:
        state_in = {"query": query, "retrieved": "", "summary": "", "sources": []}
        state_out = PDF_GRAPH.invoke(state_in)
    except Exception as e:
        return None, f"Orchestrator failed: {e}"

    summary = state_out.get("summary") or ""
    if not summary:
        return None, "No answer was generated from the PDF."

    return summary, None


def analyze_tabular(file_path: str, query: str):
    """
    Handle .xlsx, .xls, .csv as tables and send FULL data to summarizer.
    """
    if not query or not query.strip():
        return None, "Query cannot be empty."

    try:
        lower = file_path.lower()
        if lower.endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
    except Exception as e:
        return None, f"Failed to read tabular file: {e}"

    csv_data = df.to_csv(index=False)

    text = (
        "You are analysing the following tabular data (in CSV format).\n"
        "Use ALL of the data you see below to answer the user's question as accurately as possible.\n"
        "Do NOT say that you only see a sample. Just answer based on the table.\n\n"
        f"User question:\n{query}\n\n"
        f"Table:\n{csv_data}"
    )

    try:
        res = requests.post(
            f"{MCP_SERVER}/invoke/summarizer",
            json={
                "text": text,
                "style": "precise, analytical, and directly answer the question"
            },
            timeout=90,
        )
        res.raise_for_status()
        out = res.json()
        summary = out.get("summary") or ""
        if not summary:
            return None, "No answer could be generated from the tabular file."
        return summary, None
    except Exception as e:
        return None, f"Failed to analyze tabular file with LLM: {e}"


def analyze_text_file(file_path: str, query: str):
    """
    Handle .txt and other plain-text-like files with the summarizer.
    """
    if not query or not query.strip():
        return None, "Query cannot be empty."

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text_content = f.read()
    except Exception as e:
        return None, f"Failed to read text file: {e}"

    text = (
        "You are analysing the following text document.\n"
        "Answer the user's question using only this content.\n\n"
        f"User question:\n{query}\n\n"
        f"Document:\n{text_content}"
    )

    try:
        res = requests.post(
            f"{MCP_SERVER}/invoke/summarizer",
            json={"text": text, "style": "concise and accurate"},
            timeout=90,
        )
        res.raise_for_status()
        out = res.json()
        summary = out.get("summary") or ""
        if not summary:
            return None, "No answer could be generated from the text file."
        return summary, None
    except Exception as e:
        return None, f"Failed to analyze text file with LLM: {e}"


def analyze_file_path(path: str, query: str, is_new_upload: bool = False):
    """
    Route analysis based on file extension.
    - PDF -> RAG
    - Excel/CSV -> tabular analysis
    - TXT -> text analysis
    - Others -> unsupported message
    """
    if not path:
        return None, "File path is missing."
    if not query or not query.strip():
        return None, "Query cannot be empty."

    lower = path.lower()

    if lower.endswith(".pdf"):
        return run_pdf_rag(path, query, is_new_upload=is_new_upload)
    elif lower.endswith((".xlsx", ".xls", ".csv")):
        return analyze_tabular(path, query)
    elif lower.endswith((".txt", ".md")):
        return analyze_text_file(path, query)
    else:
        return None, f"Unsupported file type for path {path}. Please upload a PDF, Excel, CSV, or text file."


# ---------- Routes ----------

@app.route("/", methods=["GET", "POST"])
def index():
    """
    Home page:
      - GET: show clean dashboard (no previous summary)
      - POST: if JS fails, run once and show result
    """
    if request.method == "POST":
        uploaded = request.files.get("file")
        query = (request.form.get("query") or "").strip()

        is_new_upload = False
        file_path = None
        error = None
        summary = None

        if uploaded and uploaded.filename.strip():
            os.makedirs("uploads", exist_ok=True)
            save_path = os.path.join("uploads", uploaded.filename)
            uploaded.save(save_path)

            # Keep this file for follow-up questions
            session["active_file_path"] = save_path
            file_path = save_path
            is_new_upload = True

            lower = save_path.lower()
            # Excel/CSV drive visuals
            if lower.endswith((".xlsx", ".xls", ".csv")):
                session["file_path"] = save_path
            else:
                session.pop("file_path", None)
        else:
            file_path = session.get("active_file_path")

        if not query:
            error = "Query cannot be empty."
        elif not file_path:
            error = "Please upload a file first."
        else:
            summary, error = analyze_file_path(file_path, query, is_new_upload=is_new_upload)

        return render_template("dashboard.html", summary=summary, error=error)

    # GET request: clean page (no summary/chat)
    return render_template("dashboard.html")


@app.route("/upload", methods=["GET", "POST"])
def upload():
    """
    Used mainly for Excel/CSV -> Visuals flow via navbar.
    """
    if request.method == "GET":
        return render_template("dashboard.html")

    file = request.files.get("file")
    if not file or file.filename.strip() == "":
        return render_template("dashboard.html", error="Please upload a file.")

    os.makedirs("uploads", exist_ok=True)
    save_path = os.path.join("uploads", file.filename)
    file.save(save_path)

    # Always make it the active file
    session["active_file_path"] = save_path

    lower = save_path.lower()
    # Only Excel/CSV should be used for visuals
    if lower.endswith((".xlsx", ".xls", ".csv")):
        session["file_path"] = save_path
        return redirect(url_for("visuals"))
    else:
        session.pop("file_path", None)
        return render_template("dashboard.html", error="Uploaded file is not an Excel/CSV file for visuals.")


@app.route("/visuals", methods=["GET"])
def visuals():
    file_path = session.get("file_path")
    if not file_path:
        # If no explicit visuals file, try the active file (if it's tabular)
        active = session.get("active_file_path")
        if active and active.lower().endswith((".xlsx", ".xls", ".csv")):
            file_path = active
            session["file_path"] = active
        else:
            return render_template("dashboard.html", error="No Excel/CSV file uploaded for visuals.")

    # Get available columns only (from Excel/CSV tool)
    payload = {"file_path": file_path, "suggestions_only": True}
    meta = requests.post(
        f"{MCP_SERVER}/invoke/excel_visualizer",
        json=payload,
        timeout=60,
    ).json()
    available = meta.get("available_columns", [])

    return render_template(
        "visuals.html",
        file_path=file_path,
        available_columns=available,
        visuals=None,
    )


@app.route("/generate_visuals", methods=["POST"])
def generate_visuals():
    file_path = request.form.get("file_path") or session.get("file_path")
    x_col = request.form.get("x_column")

    if not file_path:
        return render_template(
            "visuals.html",
            error="No Excel/CSV file found for visuals.",
            file_path=None,
            available_columns=[],
            visuals=None,
        )

    if not x_col:
        # Re-fetch available duplicate columns to re-render selector
        payload = {"file_path": file_path, "suggestions_only": True}
        meta = requests.post(
            f"{MCP_SERVER}/invoke/excel_visualizer",
            json=payload,
            timeout=60,
        ).json()
        available = meta.get("available_columns", [])
        return render_template(
            "visuals.html",
            error="Please select a column.",
            file_path=file_path,
            available_columns=available,
            visuals=None,
        )

    payload = {"file_path": file_path, "x_column": x_col}
    visuals = requests.post(
        f"{MCP_SERVER}/invoke/excel_visualizer",
        json=payload,
        timeout=90,
    ).json()

    # Ask again for available columns
    payload = {"file_path": file_path, "suggestions_only": True}
    meta = requests.post(
        f"{MCP_SERVER}/invoke/excel_visualizer",
        json=payload,
        timeout=60,
    ).json()
    available = meta.get("available_columns", [])

    return render_template(
        "visuals.html",
        file_path=file_path,
        available_columns=available,
        visuals=visuals,
    )


# ---------- Chat interface (Ask Question page) ----------

@app.route("/query", methods=["GET"])
def query():
    """
    Chat-like interface for follow-up questions on the current active file.
    On refresh (GET), we do NOT load previous chat history.
    """
    return render_template("chat.html", history=[], error=None)


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """
    AJAX endpoint for chat interface.
    Uses the current active file and RAG/Tabular/Text logic to answer.
    """
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "Please enter a question."}), 400

    file_path = session.get("active_file_path")
    if not file_path:
        return jsonify({"error": "Please upload a file on Home first."}), 400

    answer, err = analyze_file_path(file_path, question, is_new_upload=False)
    if err:
        return jsonify({"error": err}), 500

    return jsonify({"answer": answer})


@app.route("/summarize", methods=["GET", "POST"])
def summarize():
    """
    Separate summarize page (optional). No persistence across refresh.
    """
    summary = None
    error = None
    file_path = session.get("active_file_path")

    if request.method == "POST":
        uploaded = request.files.get("file")
        query = (request.form.get("query") or "").strip()

        is_new_upload = False
        if uploaded and uploaded.filename.strip():
            os.makedirs("uploads", exist_ok=True)
            save_path = os.path.join("uploads", uploaded.filename)
            uploaded.save(save_path)
            session["active_file_path"] = save_path
            file_path = save_path
            is_new_upload = True

            lower = save_path.lower()
            # Excel/CSV drives visuals
            if lower.endswith((".xlsx", ".xls", ".csv")):
                session["file_path"] = save_path
            else:
                session.pop("file_path", None)
        else:
            file_path = session.get("active_file_path")

        if not file_path:
            error = "Please upload a file first."
        else:
            summary, error = analyze_file_path(file_path, query, is_new_upload=is_new_upload)

    return render_template("summarize.html", summary=summary, error=error, file_path=file_path)


@app.route("/api/summarize", methods=["POST"])
def api_summarize():
    """
    AJAX endpoint used by dashboard.js from the Home page.
    """
    uploaded = request.files.get("file")
    query = (request.form.get("query") or "").strip()

    if not query:
        return jsonify({"error": "Query cannot be empty."})

    is_new_upload = False
    file_path = None

    if uploaded and uploaded.filename.strip():
        os.makedirs("uploads", exist_ok=True)
        save_path = os.path.join("uploads", uploaded.filename)
        uploaded.save(save_path)
        session["active_file_path"] = save_path
        file_path = save_path
        is_new_upload = True

        lower = save_path.lower()
        # Excel/CSV drives visuals; clear any old Excel if non-tabular uploaded
        if lower.endswith((".xlsx", ".xls", ".csv")):
            session["file_path"] = save_path
        else:
            session.pop("file_path", None)
    else:
        file_path = session.get("active_file_path")

    if not file_path:
        return jsonify({"error": "Please upload a file first."})

    summary, error = analyze_file_path(file_path, query, is_new_upload=is_new_upload)
    if error:
        return jsonify({"error": error})

    return jsonify({"summary": summary})
# ===================================
# === ML EXTENSION: ROUTE START   ===
# ===================================

@app.route("/ml", methods=["GET", "POST"])
def ml():
    """
    ML page:

    - Uses the current Excel/CSV file as historical data.
    - Train: choose a target column, train a model via MCP.
    - Predict: user fills a form (one input per feature), and we validate types
      (numeric vs text) BEFORE calling the ML tool.

    To remove ML:
      - Delete this route.
      - Remove ml.html template.
      - Remove ML imports and registration.
    """
    # Use the current tabular file (Excel/CSV)
    file_path = session.get("file_path") or session.get("active_file_path")
    if not file_path or not file_path.lower().endswith((".xlsx", ".xls", ".csv")):
        return render_template(
            "ml.html",
            error="Please upload an Excel or CSV file on Home/Upload first.",
            columns=[],
            model_id=session.get("ml_model_id"),
            train_text=None,
            predict_text=None,
            predict_result=None,
            feature_schema=None,
        )

    # Read file and get columns + dtypes
    try:
        lower = file_path.lower()
        if lower.endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        columns = list(df.columns)
    except Exception as e:
        return render_template(
            "ml.html",
            error=f"Failed to read file for ML: {e}",
            columns=[],
            model_id=session.get("ml_model_id"),
            train_text=None,
            predict_text=None,
            predict_result=None,
            feature_schema=None,
        )

    error = None
    train_text = None
    predict_text = None
    predict_result = None
    model_id = session.get("ml_model_id")
    feature_schema = session.get("ml_schema")  # { "target": "...", "features": {col: "numeric"/"text"} }

    if request.method == "POST":
        action = request.form.get("action")

        # ---- Train model ----
        if action == "train":
            target_col = request.form.get("target_column")
            if not target_col:
                error = "Please select a target column to train."
            elif target_col not in columns:
                error = f"Target column '{target_col}' not found."
            else:
                payload = {
                    "mode": "train",
                    "file_path": file_path,
                    "target_column": target_col,
                }
                try:
                    res = requests.post(
                        f"{MCP_SERVER}/invoke/ml_modeler",
                        json=payload,
                        timeout=120,
                    )
                    out = res.json()
                except Exception as e:
                    error = f"ML training failed: {e}"
                    out = {}

                if "error" in out:
                    error = out["error"]
                else:
                    model_id = out.get("model_id")
                    session["ml_model_id"] = model_id

                    # Short plain-text training summary
                    is_reg = out.get("is_regression", False)
                    score = out.get("score", None)
                    tcol = out.get("target_column", target_col)
                    if is_reg:
                        ttype = "Regression"
                    else:
                        ttype = "Classification"

                    if score is not None:
                        train_text = f"Trained {ttype} model for '{tcol}' (score: {score:.3f})."
                    else:
                        train_text = f"Trained {ttype} model for '{tcol}'."

                    # Build feature schema (for prediction form)
                    features = {}
                    for col in columns:
                        if col == target_col:
                            continue
                        if pd.api.types.is_numeric_dtype(df[col]):
                            features[col] = "numeric"
                        else:
                            features[col] = "text"

                    feature_schema = {
                        "target": target_col,
                        "features": features,
                    }
                    session["ml_schema"] = feature_schema

        # ---- Predict using trained model ----
        elif action == "predict":
            if not model_id:
                error = "No trained model found. Train a model first."
            else:
                feature_schema = session.get("ml_schema")
                if not feature_schema:
                    error = "No feature schema found. Train a model again."
                else:
                    features = feature_schema.get("features", {})
                    input_dict = {}
                    # Validate each feature according to its type
                    for col_name, ftype in features.items():
                        form_key = f"feat_{col_name}"
                        raw_val = request.form.get(form_key, "").strip()

                        if raw_val == "":
                            error = f"Please enter a value for '{col_name}'."
                            break

                        if ftype == "numeric":
                            try:
                                # Accept integers or floats
                                if "." in raw_val:
                                    val = float(raw_val)
                                else:
                                    # try int, fallback float
                                    try:
                                        val = int(raw_val)
                                    except ValueError:
                                        val = float(raw_val)
                            except ValueError:
                                error = f"'{col_name}' must be a number."
                                break
                            input_dict[col_name] = val
                        else:
                            # text / categorical
                            input_dict[col_name] = raw_val

                    if not error:
                        payload = {
                            "mode": "predict",
                            "model_id": model_id,
                            "input": input_dict,
                        }
                        try:
                            res = requests.post(
                                f"{MCP_SERVER}/invoke/ml_modeler",
                                json=payload,
                                timeout=60,
                            )
                            out = res.json()
                        except Exception as e:
                            error = f"ML prediction failed: {e}"
                            out = {}

                        if "error" in out:
                            error = out["error"]
                        else:
                            predict_result = out

                            # Short plain-text prediction
                            if out.get("is_regression"):
                                # Regression: numeric target
                                tcol = out.get("target_column", "Target")
                                val = out.get("prediction", None)
                                if val is not None:
                                    predict_text = f"{tcol}: {round(val, 2)}"
                                else:
                                    predict_text = f"{tcol}: (no prediction)"
                            else:
                                # Classification: label + confidence if available
                                tcol = out.get("target_column", "Target")
                                label = out.get("predicted_class", "Unknown")
                                probs = out.get("probabilities")
                                if probs:
                                    best = max(probs)
                                    pct = round(best * 100, 1)
                                    predict_text = f"{tcol}: {label} ({pct}%)"
                                else:
                                    predict_text = f"{tcol}: {label}"

    return render_template(
        "ml.html",
        error=error,
        columns=columns,
        model_id=model_id,
        train_text=train_text,
        predict_text=predict_text,
        predict_result=predict_result,
        feature_schema=feature_schema,
    )

# =================================
# === ML EXTENSION: ROUTE END   ===
# =================================
# ======================================
# === REQ→TC EXTENSION: ROUTE START ===
# ======================================

@app.route("/req2tc", methods=["GET", "POST"])
def req2tc():
    """
    Requirement → Testcase page.

    Generic behavior:
      - Upload an Excel/CSV with at least:
          - 'Requirement' column
          - One or more additional columns describing each testcase
            (e.g. TC_ID, Module, Type, Title, Preconditions, Steps, Expected, ...)
      - For each row, we build a plain-text testcase block using ALL columns except 'Requirement':
            ColumnName: value
            ColumnName2: value2
        Then we group all testcases per Requirement and index them.

      - When generating, the LLM sees these examples and is instructed to
        follow the SAME field names and structure in its output.

    To remove this feature:
      - Remove this route
      - Delete req2tc.html
      - Remove req2tc_tool from mcp_server.py
    """
    status_message = None
    error = None
    generated = None

    if request.method == "POST":
        action = request.form.get("action")

        # --- 1) Upload & index old requirements + testcases ---
        if action == "index":
            mapping_file = request.files.get("mapping_file")
            if not mapping_file or mapping_file.filename.strip() == "":
                error = "Please upload a mapping file (Excel or CSV)."
            else:
                os.makedirs("uploads", exist_ok=True)
                save_path = os.path.join("uploads", mapping_file.filename)
                mapping_file.save(save_path)

                # Read the mapping file
                try:
                    lower = save_path.lower()
                    if lower.endswith(".csv"):
                        df = pd.read_csv(save_path)
                    else:
                        df = pd.read_excel(save_path)
                except Exception as e:
                    error = f"Failed to read mapping file: {e}"
                    df = None

                if df is not None and not error:
                    columns = list(df.columns)
                    if "Requirement" not in columns:
                        error = "Mapping file must contain a 'Requirement' column."
                    else:
                        # All other columns are considered testcase attributes
                        testcase_cols = [c for c in columns if c != "Requirement"]
                        if not testcase_cols:
                            error = (
                                "Mapping file must have at least one additional column "
                                "besides 'Requirement' to describe testcases."
                            )

                    if df is not None and not error:
                        # Group testcases by requirement
                        entries = []
                        grouped = df.groupby("Requirement", dropna=True)
                        for req_text, group in grouped:
                            req_str = str(req_text)
                            tc_blocks = []
                            for _, row in group.iterrows():
                                lines = []
                                for col in testcase_cols:
                                    val = row.get(col)
                                    if pd.isna(val):
                                        continue
                                    val_str = str(val).strip()
                                    if val_str == "":
                                        continue
                                    # Generic "FieldName: value" format
                                    lines.append(f"{col}: {val_str}")
                                if lines:
                                    tc_blocks.append("\n".join(lines))
                            if not tc_blocks:
                                continue
                            all_tcs = "\n\n".join(tc_blocks)
                            entries.append(
                                {
                                    "requirement": req_str,
                                    "testcases": all_tcs,
                                }
                            )

                        if not entries:
                            error = "No valid requirement/testcase rows found in mapping file."
                        else:
                            # Send to MCP tool for indexing
                            try:
                                res = requests.post(
                                    f"{MCP_SERVER}/invoke/req2tc_generator",
                                    json={"mode": "index", "entries": entries},
                                    timeout=180,
                                )
                                out = res.json()
                            except Exception as e:
                                error = f"Indexing failed: {e}"
                                out = {}

                            if not error:
                                if "error" in out:
                                    error = out["error"]
                                else:
                                    indexed_count = out.get("indexed_count", 0)
                                    status_message = (
                                        f"Indexed {indexed_count} requirements from mapping file."
                                    )

        # --- 2) Generate test cases for a new requirement ---
        elif action == "generate":
            new_req = (request.form.get("new_requirement") or "").strip()
            if not new_req:
                error = "Please enter a new requirement."
            else:
                try:
                    res = requests.post(
                        f"{MCP_SERVER}/invoke/req2tc_generator",
                        json={"mode": "generate", "requirement": new_req},
                        timeout=120,
                    )
                    out = res.json()
                except Exception as e:
                    error = f"Generation failed: {e}"
                    out = {}

                if not error:
                    if "error" in out:
                        error = out["error"]
                    else:
                        generated = out.get("testcases", "")

    return render_template(
        "req2tc.html",
        error=error,
        status_message=status_message,
        generated=generated,
    )

# ====================================
# === REQ→TC EXTENSION: ROUTE END ===
# ====================================
# =========================================
# === EVENT RISK EXTENSION: ROUTE START ===
# =========================================

@app.route("/event_risk", methods=["GET", "POST"])
def event_risk():
    """
    Event risk forecasting page.

    - Uses the current uploaded CSV/Excel file as log data.
    - Step 1: Index the log file via MCP risk_forecaster (mode='index').
    - Step 2: Choose dimension (severity/service/error_code/message), value, and window (day/week/month).
              The backend calls risk_forecaster (mode='forecast') and shows:
                - expected_count
                - probability_at_least_one
    """
    file_path = session.get("file_path") or session.get("active_file_path")
    error = None
    status_message = None
    forecast = None
    options = {
        "severity": [],
        "service": [],
        "error_code": [],
        "message": [],
    }

    # Load dimension values if we have a log file
    df = None
    if file_path and file_path.lower().endswith((".csv", ".xlsx", ".xls")):
        try:
            lower = file_path.lower()
            if lower.endswith(".csv"):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)

            for col in options.keys():
                if col in df.columns:
                    vals = (
                        df[col]
                        .dropna()
                        .astype(str)
                        .value_counts()
                        .index.tolist()
                    )
                    # Limit to top 50 to keep dropdown manageable
                    options[col] = vals[:50]
        except Exception as e:
            error = f"Failed to read log file: {e}"
    else:
        if request.method == "GET":
            error = "Please upload a CSV/Excel log file on the Home page first."

    if request.method == "POST":
        action = request.form.get("action")

        # --- Step 1: Index logs in MCP ---
        if action == "index":
            if not file_path:
                error = "No file found in session. Upload a log file first."
            else:
                try:
                    res = requests.post(
                        f"{MCP_SERVER}/invoke/risk_forecaster",
                        json={"mode": "index", "file_path": file_path},
                        timeout=180,
                    )
                    out = res.json()
                except Exception as e:
                    error = f"Indexing failed: {e}"
                    out = {}

                if not error:
                    if "error" in out:
                        error = out["error"]
                    else:
                        row_count = out.get("row_count", 0)
                        status_message = f"Indexed {row_count} log rows for risk forecasting."

        # --- Step 2: Forecast risk ---
        elif action == "forecast":
            dimension = request.form.get("dimension")
            value = request.form.get("value")
            window = request.form.get("window")

            if not dimension or not value or not window:
                error = "Please select dimension, value, and window."
            else:
                payload = {
                    "mode": "forecast",
                    "dimension": dimension,
                    "value": value,
                    "window": window,
                }
                try:
                    res = requests.post(
                        f"{MCP_SERVER}/invoke/risk_forecaster",
                        json=payload,
                        timeout=60,
                    )
                    out = res.json()
                except Exception as e:
                    error = f"Forecasting failed: {e}"
                    out = {}

                if not error:
                    if "error" in out:
                        error = out["error"]
                    else:
                        forecast = out

    return render_template(
        "event_risk.html",
        error=error,
        status_message=status_message,
        forecast=forecast,
        options=options,
        file_path=file_path,
    )

# =======================================
# === EVENT RISK EXTENSION: ROUTE END ===
# =======================================



if __name__ == "__main__":
    app.run(debug=True)
