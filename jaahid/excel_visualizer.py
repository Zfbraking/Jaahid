import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import base64
import json
from datetime import datetime


def build_excel_tool(llm_fn=None):

    class ExcelTool:
        name = "excel_visualizer"
        description = "Generate workstream-based KPI and charts using LLM to filter unnecessary columns."

        def __init__(self, llm_fn=None):
            self.llm_fn = llm_fn

        # ---------------------------------------------------
        # MAIN INVOKE
        # ---------------------------------------------------
        def invoke(self, payload):
            file_path = payload.get("file_path")
            if not file_path:
                return {"error": "file_path is required"}

            # Load file
            try:
                if file_path.lower().endswith((".csv", ".txt")):
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path)
            except Exception as e:
                return {"error": f"Failed to read file: {e}"}

            if "Workstream" not in df.columns:
                return {"error": "The file must contain a 'Workstream' column."}

            # ----- Compute global KPIs -----
            kpis = self._compute_kpis(df)

            # ----- Decide which columns to visualize -----
            selected_columns = self._select_visual_columns(df)

            if not selected_columns:
                return {
                    "error": "No meaningful columns selected for charts.",
                    "kpis": kpis,
                    "charts": [],
                }

            charts = []

            # Per workstream visuals
            workstreams = df["Workstream"].dropna().unique()

            for ws in workstreams:
                ws_df = df[df["Workstream"] == ws]

                ws_block = {
                    "workstream": ws,
                    "charts": []   # one chart block per column
                }

                for col in selected_columns:
                    if col not in ws_df.columns:
                        continue
                    chart_obj = self._make_chart_for_column(ws_df, col, ws)
                    if chart_obj:
                        ws_block["charts"].append(chart_obj)

                charts.append(ws_block)

            return {"kpis": kpis, "charts": charts}

        # ---------------------------------------------------
        # KPI COMPUTATION
        # ---------------------------------------------------
        def _compute_kpis(self, df: pd.DataFrame):
            total = len(df)

            # Status-based KPIs
            status_col = None
            for c in df.columns:
                if c.lower() == "status":
                    status_col = c
                    break

            completed = in_progress = blocked = not_started = 0

            if status_col:
                s = df[status_col].astype(str).str.lower()

                completed = s.str.contains("compl|done").sum()
                in_progress = s.str.contains("progress|in progress|ongoing|working").sum()
                blocked = s.str.contains("block|blocked|hold|on hold|waiting").sum()
                not_started = s.str.contains("not started|new|todo").sum()

            completion_percent = (completed / total * 100) if total else 0.0

            # Overdue: based on End Date if present
            overdue = 0
            end_col = None
            for c in df.columns:
                if "end date" in c.lower() or "finish" in c.lower() or "due" in c.lower():
                    end_col = c
                    break

            if end_col:
                try:
                    end_dates = pd.to_datetime(df[end_col], errors="coerce")
                    today = pd.Timestamp(datetime.now().date())
                    overdue = (end_dates < today).sum()
                except Exception:
                    overdue = 0

            # Complexity: use existing column if present (Low/Medium/High)
            complexity_col = None
            for c in df.columns:
                if "complexity" in c.lower():
                    complexity_col = c
                    break

            avg_complexity_score = None
            avg_complexity_label = None

            if complexity_col:
                mapping = {"low": 1, "medium": 2, "med": 2, "high": 3}
                vals = df[complexity_col].astype(str).str.lower().map(mapping)
                vals = vals.dropna()
                if len(vals) > 0:
                    avg_complexity_score = vals.mean()
                    if avg_complexity_score < 1.5:
                        avg_complexity_label = "Low"
                    elif avg_complexity_score < 2.5:
                        avg_complexity_label = "Medium"
                    else:
                        avg_complexity_label = "High"

            return {
                "total": int(total),
                "completed": int(completed),
                "in_progress": int(in_progress),
                "blocked": int(blocked),
                "not_started": int(not_started),
                "completion_percent": float(completion_percent),
                "overdue": int(overdue),
                "avg_complexity_score": float(avg_complexity_score) if avg_complexity_score is not None else None,
                "avg_complexity_label": avg_complexity_label,
            }

        # ---------------------------------------------------
        # LLM COLUMN SELECTION
        # ---------------------------------------------------
        def _select_visual_columns(self, df: pd.DataFrame):
            candidates = []
            metadata = []

            for col in df.columns:
                cname = col.lower()

                # Always skip these
                if cname in ["workstream", "owner", "comments"]:
                    continue

                # Skip date-like columns
                if "date" in cname:
                    continue

                nunique = df[col].nunique(dropna=True)

                # Skip super high-cardinality free text
                if df[col].dtype == "object" and nunique > 50:
                    continue

                # Build metadata for LLM
                values = df[col].dropna().unique()
                sample = [str(v) for v in values[:5]]

                metadata.append({
                    "name": col,
                    "dtype": str(df[col].dtype),
                    "unique_values": int(nunique),
                    "sample_values": sample
                })
                candidates.append(col)

            # Fallback if no LLM
            if not self.llm_fn:
                defaults = ["Status", "Environment", "Complexity", "Migration Wave"]
                return [c for c in defaults if c in df.columns]

            prompt = f"""
You are selecting meaningful columns for summary charts on an IT Transition dashboard.

We want only a few high-level categorical columns that make sense to visualize:
- Status (workflow state)
- Environment (DEV/QA/UAT/PROD)
- Complexity (Low/Medium/High)
- Migration wave / phase
- Any similar categorical with <= 20 unique values.

We want to avoid:
- Owner names
- Comments / free text
- Date fields
- IDs
- Very high-cardinality fields

Here is metadata of candidate columns:
{json.dumps(metadata, indent=2)}

Return ONLY a JSON array of column names to visualize, e.g.:
["Status", "Environment", "Complexity"]

Your response MUST be pure JSON. No extra text.
"""

            try:
                raw = self.llm_fn(prompt).strip()

                # Handle ```json ... ``` style responses
                if raw.startswith("```"):
                    raw = raw.strip("`")
                    if raw.lower().startswith("json"):
                        raw = raw[4:].strip()

                selected = json.loads(raw)
                selected = [c for c in selected if c in df.columns]

                if selected:
                    return selected
            except Exception:
                pass

            # Fallback if LLM fails
            fallback = ["Status", "Environment", "Complexity", "Migration Wave"]
            return [c for c in fallback if c in df.columns]

        # ---------------------------------------------------
        # CHART BUILDER (ONE TYPE PER COLUMN)
        # ---------------------------------------------------
        def _make_chart_for_column(self, df, col, ws_name):
            series = df[col]
            counts = series.value_counts(dropna=False)

            if counts.empty:
                return None

            # Normalize index to strings
            counts.index = counts.index.map(lambda x: "NaN" if pd.isna(x) else str(x))

            cname = col.lower()

            # Decide chart type:
            # - Status -> Pie
            # - Migration Wave -> Line
            # - Else -> Bar
            if "status" in cname or "rag" in cname:
                chart_type = "pie"
            elif "wave" in cname:
                chart_type = "line"
            else:
                chart_type = "bar"

            title = f"{ws_name} – {col}"

            fig, ax = plt.subplots()

            if chart_type == "pie":
                counts.plot(kind="pie", ax=ax, autopct="%1.1f%%")
                ax.set_ylabel("")
            elif chart_type == "line":
                line_series = counts.sort_index()
                line_series.plot(kind="line", marker="o", ax=ax)
                ax.grid(True, linestyle="--", alpha=0.5)
            else:
                counts.plot(kind="bar", ax=ax)

            ax.set_title(title)
            ax.set_xlabel(col)
            if chart_type != "pie":
                ax.set_ylabel("Count")

            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            plt.close(fig)
            buf.seek(0)
            chart_b64 = base64.b64encode(buf.read()).decode("utf-8")

            return {
                "column": col,
                "chart_type": chart_type,
                "title": title,
                "chart": chart_b64,
            }

    return ExcelTool(llm_fn=llm_fn)
