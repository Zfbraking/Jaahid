import os
from flask import Flask, render_template, request

from excel_visualizer import build_excel_tool
from llm_setup import llm
from agents.orchestrator import SmartRiskOrchestrator

UPLOAD_FOLDER = "uploaded_files"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)

# Existing generic visualizer (home page)
excel_tool = build_excel_tool(llm_fn=llm)

# Risk-focused orchestrator (smart risk view)
risk_orchestrator = SmartRiskOrchestrator()


# ---------- MAIN DASHBOARD (existing style) ----------
@app.route("/", methods=["GET", "POST"])
def dashboard():
    charts = None
    error = None
    file_name = None
    kpis = None  # if your excel_tool returns kpis; otherwise unused

    if request.method == "POST":
        uploaded_file = request.files.get("file")
        if not uploaded_file:
            error = "Please upload an Excel/CSV file."
        else:
            file_name = uploaded_file.filename
            save_path = os.path.join(UPLOAD_FOLDER, file_name)
            uploaded_file.save(save_path)

            result = excel_tool.invoke({"file_path": save_path})

            if "error" in result and result.get("charts") is None:
                error = result["error"]
            else:
                charts = result.get("charts")
                kpis = result.get("kpis")
                if "error" in result and not error:
                    error = result["error"]

    return render_template(
        "dashboard.html",
        charts=charts,
        kpis=kpis,
        error=error,
        file_name=file_name
    )


# ---------- RISK & CRITICALITY DASHBOARD ----------
@app.route("/smart", methods=["GET", "POST"])
def smart_dashboard():
    charts = None
    error = None
    file_name = None
    kpis = None
    group_column = None
    app_risk_breakdown = None
    global_charts = None

    if request.method == "POST":
        uploaded_file = request.files.get("file")
        if not uploaded_file:
            error = "Please upload an Excel/CSV file."
        else:
            file_name = uploaded_file.filename
            save_path = os.path.join(UPLOAD_FOLDER, file_name)
            uploaded_file.save(save_path)

            result = risk_orchestrator.analyze_file(save_path)

            group_column = result.get("group_column")
            kpis = result.get("kpis")
            charts = result.get("charts")
            app_risk_breakdown = result.get("app_risk_breakdown")
            global_charts = result.get("global_charts")

            if "error" in result and charts is None and not app_risk_breakdown:
                error = result["error"]
            elif "error" in result and not error:
                error = result["error"]

    return render_template(
        "smart_dashboard.html",
        charts=charts,
        kpis=kpis,
        app_risk_breakdown=app_risk_breakdown,
        global_charts=global_charts,
        group_column=group_column,
        error=error,
        file_name=file_name
    )


if __name__ == "__main__":
    app.run(port=8000, debug=True)
