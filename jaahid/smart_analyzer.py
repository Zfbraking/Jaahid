import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import base64
import json
from datetime import datetime
from typing import Dict, Any, Optional, List

from llm_setup import llm  # your existing LLM setup


class SmartTransitionAnalyzer:
    """
    Generic, LLM-driven Excel analyzer:
    - Detects grouping column (e.g. Workstream / Tower / Domain / Project)
    - Detects status/end-date/complexity columns for KPIs
    - Lets LLM choose which columns to visualize
    - Builds chart images in base64
    """

    def __init__(self, llm_fn=llm):
        self.llm_fn = llm_fn

    # ---------- Public entry ----------
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        try:
            if file_path.lower().endswith((".csv", ".txt")):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
        except Exception as e:
            return {"error": f"Failed to read file: {e}"}

        if df.empty or len(df.columns) == 0:
            return {"error": "The file is empty or has no columns."}

        # 1) detect grouping column
        group_col = self._detect_group_column(df)
        if not group_col:
            return {"error": "Could not detect a suitable grouping column for charts."}

        # 2) detect semantic columns (status, end date, complexity)
        semantic_cols = self._detect_semantic_columns(df)

        # 3) compute KPIs
        kpis = self._compute_kpis(df, semantic_cols)

        # 4) decide which columns to visualize
        visual_cols = self._select_visual_columns(df, group_col)
        if not visual_cols:
            return {
                "error": "No meaningful columns selected for charts.",
                "group_column": group_col,
                "kpis": kpis,
                "charts": [],
            }

        # 5) build charts per group value
        charts = []
        for group_value, gdf in df.groupby(group_col):
            block = {
                "group_value": str(group_value),
                "charts": []
            }
            for col in visual_cols:
                if col not in gdf.columns:
                    continue
                chart_obj = self._make_chart_for_column(gdf, col, group_value)
                if chart_obj:
                    block["charts"].append(chart_obj)
            charts.append(block)

        return {
            "group_column": group_col,
            "kpis": kpis,
            "charts": charts
        }

    # ---------- Group column detection ----------
    def _detect_group_column(self, df: pd.DataFrame) -> Optional[str]:
        candidates = []
        metadata = []

        for col in df.columns:
            nunique = df[col].nunique(dropna=True)
            if 2 <= nunique <= 25:
                candidates.append(col)
                metadata.append({
                    "name": col,
                    "unique_values": int(nunique),
                    "sample_values": [str(v) for v in df[col].dropna().unique()[:5]]
                })

        if not candidates:
            return None

        # simple heuristic if LLM not available
        if not self.llm_fn:
            preferred = ["workstream", "tower", "domain", "area", "project", "program", "application", "segment"]
            for col in candidates:
                cname = col.lower()
                if any(k in cname for k in preferred):
                    return col
            return candidates[0]

        prompt = f"""
You are analyzing a generic Excel file used for IT or business transition tracking.

We want to choose ONE column that is best suited as a high-level grouping dimension
for dashboards – similar to "Workstream", "Tower", "Domain", "Application Group", "Project", or "Business Area".

We have these candidate columns with low unique values:
{json.dumps(metadata, indent=2)}

Return ONLY the name of the column (a single JSON string) that is the best choice to group charts.
Example:
"Workstream"

Your response MUST be a pure JSON string, e.g. "Tower"
"""
        try:
            raw = self.llm_fn(prompt).strip()
            if raw.startswith("```"):
                raw = raw.strip("`")
                if raw.lower().startswith("json"):
                    raw = raw[4:].strip()
            col_name = json.loads(raw)
            if isinstance(col_name, str) and col_name in df.columns:
                return col_name
        except Exception:
            pass

        return candidates[0]

    # ---------- Semantic column detection (status, end date, complexity) ----------
    def _detect_semantic_columns(self, df: pd.DataFrame) -> Dict[str, Optional[str]]:
        metadata = []
        for col in df.columns:
            nunique = df[col].nunique(dropna=True)
            metadata.append({
                "name": col,
                "dtype": str(df[col].dtype),
                "unique_values": int(nunique),
                "sample_values": [str(v) for v in df[col].dropna().unique()[:5]]
            })

        # heuristic defaults
        status_col = None
        end_date_col = None
        complexity_col = None

        for col in df.columns:
            cname = col.lower()
            if status_col is None and any(k in cname for k in ["status", "state", "progress", "stage"]):
                status_col = col
            if end_date_col is None and any(k in cname for k in ["end date", "finish", "due", "target date"]):
                end_date_col = col
            if complexity_col is None and any(k in cname for k in ["complexity", "severity", "priority"]):
                complexity_col = col

        if not self.llm_fn:
            return {"status": status_col, "end_date": end_date_col, "complexity": complexity_col}

        prompt = f"""
You are analyzing columns of a transition/progress tracking Excel.

We want to identify which columns represent:
- current STATUS of work items (Not Started / In Progress / Completed / Blocked etc.)
- END DATE or target completion date
- COMPLEXITY or effort/priority (Low/Medium/High, Severity, Priority, etc.)

Here is metadata:
{json.dumps(metadata, indent=2)}

Return a JSON object like:
{{
  "status": "<column name or null>",
  "end_date": "<column name or null>",
  "complexity": "<column name or null>"
}}

Column names must exactly match metadata. Use null if uncertain.
Your response MUST be pure JSON.
"""
        try:
            raw = self.llm_fn(prompt).strip()
            if raw.startswith("```"):
                raw = raw.strip("`")
                if raw.lower().startswith("json"):
                    raw = raw[4:].strip()
            obj = json.loads(raw)

            def valid(name):
                return isinstance(name, str) and name in df.columns

            if valid(obj.get("status")):
                status_col = obj["status"]
            if valid(obj.get("end_date")):
                end_date_col = obj["end_date"]
            if valid(obj.get("complexity")):
                complexity_col = obj["complexity"]
        except Exception:
            pass

        return {"status": status_col, "end_date": end_date_col, "complexity": complexity_col}

    # ---------- KPI computation ----------
    def _compute_kpis(self, df: pd.DataFrame, sem: Dict[str, Optional[str]]) -> Dict[str, Any]:
        total = len(df)
        completed = in_progress = blocked = not_started = 0

        status_col = sem.get("status")
        if status_col and status_col in df.columns:
            s = df[status_col].astype(str).str.lower()
            completed = s.str.contains("compl|done|closed|finished").sum()
            in_progress = s.str.contains("progress|in progress|ongoing|working|running").sum()
            blocked = s.str.contains("block|blocked|hold|on hold|waiting|stuck").sum()
            not_started = s.str.contains("not started|new|todo|backlog").sum()

        completion_percent = (completed / total * 100) if total else 0.0

        overdue = 0
        end_col = sem.get("end_date")
        if end_col and end_col in df.columns:
            try:
                end_dates = pd.to_datetime(df[end_col], errors="coerce")
                today = pd.Timestamp(datetime.now().date())
                overdue = (end_dates < today).sum()
            except Exception:
                overdue = 0

        avg_complexity_score = None
        avg_complexity_label = None
        complexity_col = sem.get("complexity")
        if complexity_col and complexity_col in df.columns:
            series = df[complexity_col].astype(str).str.lower()
            mapping = {
                "low": 1, "l": 1,
                "medium": 2, "med": 2, "mid": 2,
                "high": 3, "h": 3,
                "critical": 3, "sev1": 3
            }
            scores = series.map(mapping)
            scores = scores.dropna()
            if len(scores) > 0:
                avg_complexity_score = scores.mean()
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

    # ---------- Visual column selection ----------
    def _select_visual_columns(self, df: pd.DataFrame, group_col: str) -> List[str]:
        candidates = []
        metadata = []

        for col in df.columns:
            if col == group_col:
                continue
            cname = col.lower()
            if any(k in cname for k in ["comment", "description", "note", "remark"]):
                continue
            if "id" in cname:
                continue

            nunique = df[col].nunique(dropna=True)
            if nunique == 0 or nunique > 50:
                continue

            metadata.append({
                "name": col,
                "dtype": str(df[col].dtype),
                "unique_values": int(nunique),
                "sample_values": [str(v) for v in df[col].dropna().unique()[:5]]
            })
            candidates.append(col)

        if not candidates:
            return []

        if not self.llm_fn:
            defaults = ["Status", "Environment", "Complexity", "Migration Wave", "Risk"]
            return [c for c in defaults if c in df.columns]

        prompt = f"""
We are building an executive dashboard from this Excel.

We have chosen '{group_col}' as the grouping dimension.
Now we want to choose a small set of columns to visualize as aggregated charts (counts per category) BY that grouping.

Good examples: Status, Environment, Complexity, Migration Wave, Risk Level, Phase, Region, etc.

Avoid:
- Owner names / people
- Comments / free text
- IDs
- Date fields

Here are candidate columns:
{json.dumps(metadata, indent=2)}

Return ONLY a JSON array of column names to visualize, e.g.:
["Status", "Environment", "Complexity"]

Your response MUST be pure JSON, no explanations.
"""
        try:
            raw = self.llm_fn(prompt).strip()
            if raw.startswith("```"):
                raw = raw.strip("`")
                if raw.lower().startswith("json"):
                    raw = raw[4:].strip()
            cols = json.loads(raw)
            cols = [c for c in cols if c in df.columns]
            if cols:
                return cols
        except Exception:
            pass

        fallback = ["Status", "Environment", "Complexity", "Migration Wave", "Risk"]
        return [c for c in fallback if c in df.columns]

    # ---------- Chart builder ----------
    def _make_chart_for_column(self, df_group: pd.DataFrame, col: str, group_value: str) -> Optional[Dict[str, Any]]:
        series = df_group[col]
        counts = series.value_counts(dropna=False)
        if counts.empty:
            return None

        counts.index = counts.index.map(lambda x: "NaN" if pd.isna(x) else str(x))

        cname = col.lower()
        if any(k in cname for k in ["status", "state", "rag"]):
            chart_type = "pie"
        elif any(k in cname for k in ["wave", "phase", "release"]):
            chart_type = "line"
        else:
            chart_type = "bar"

        title = f"{group_value} – {col}"

        fig, ax = plt.subplots()
        if chart_type == "pie":
            counts.plot(kind="pie", ax=ax, autopct="%1.1f%%")
            ax.set_ylabel("")
        elif chart_type == "line":
            line_series = counts.sort_index()
            line_series.plot(kind="line", marker="o", ax=ax)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.set_ylabel("Count")
        else:
            counts.plot(kind="bar", ax=ax)
            ax.set_ylabel("Count")

        ax.set_title(title)
        if chart_type != "pie":
            ax.set_xlabel(col)

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
