# agents/orchestrator.py

from typing import Dict, Any, List, Optional

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import base64

from .schema_agent import SchemaAgent
from .kpi_agent import KPIAgent
from .viz_planner_agent import VizPlannerAgent
from .chart_agent import ChartAgent


class SmartRiskOrchestrator:
    """
    Orchestrator agent for RISK & CRITICALITY view:
    - Understands structure (group + risk columns)
    - Computes risk KPIs
    - Picks only risk/criticality columns to chart (if we want them)
    - Builds Application Risk Breakdown (vertical stacked bar)
    - Builds overall charts (criticality, risk item, status)
    """

    def __init__(self):
        self.schema_agent = SchemaAgent()
        self.kpi_agent = KPIAgent()
        self.viz_agent = VizPlannerAgent()
        self.chart_agent = ChartAgent()

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

        # 1) grouping dimension (still used for semantics, but we won't show area-wise charts)
        group_col = self.schema_agent.detect_group_column(df)
        if not group_col:
            return {"error": "Could not detect a suitable grouping column for risk charts."}

        # 2) risk semantics
        sem = self.schema_agent.detect_risk_semantics(df)

        # 3) risk KPIs
        kpis = self.kpi_agent.compute_risk_kpis(df, sem)

        # 4) risk visual columns (per-group charts) – we still compute, but you don't display them
        visual_cols = self.viz_agent.select_risk_visual_columns(df, group_col)

        charts = []
        if visual_cols:
            for group_value, gdf in df.groupby(group_col):
                block = {
                    "group_value": str(group_value),
                    "charts": []
                }
                for col in visual_cols:
                    if col not in gdf.columns:
                        continue
                    chart_obj = self.chart_agent.make_chart_for_column(gdf, col, group_value)
                    if chart_obj:
                        block["charts"].append(chart_obj)
                charts.append(block)

        # 5) Application risk breakdown (for vertical stacked bar chart)
        app_risk_breakdown = self._build_application_breakdown(df, sem)

        # 6) Overall charts (criticality, risk item, status – status as PIE now)
        global_charts = self._build_global_charts(df, sem)

        return {
            "group_column": group_col,
            "kpis": kpis,
            "charts": charts,  # you won't render these area-wise charts anymore in the template
            "app_risk_breakdown": app_risk_breakdown,
            "global_charts": global_charts,
        }

    # ----- Application stacked vertical bar chart -----
    def _build_application_breakdown(
        self,
        df: pd.DataFrame,
        sem: Dict[str, Optional[str]]
    ) -> Dict[str, Any]:
        """
        Returns:
        {
          "rows": [... per-app breakdown ...],
          "chart": "<base64 png>" or None
        }
        """
        app_col = sem.get("application")
        risk_level_col = sem.get("risk_level") or sem.get("criticality")

        if not app_col or not risk_level_col:
            return {"rows": [], "chart": None}
        if app_col not in df.columns or risk_level_col not in df.columns:
            return {"rows": [], "chart": None}

        rows: List[Dict[str, Any]] = []
        for app_value, gdf in df.groupby(app_col):
            # reuse KPIAgent-style classification for this subset
            series = gdf[risk_level_col]
            high, medium, low, critical = self.kpi_agent._classify_levels(series)  # reuse logic

            total = int(high + medium + low)
            if total == 0:
                continue

            rows.append({
                "application": str(app_value),
                "high": int(high),
                "medium": int(medium),
                "low": int(low),
                "total": total,
            })

        if not rows:
            return {"rows": [], "chart": None}

        # sort by total descending, keep top 10
        rows = sorted(rows, key=lambda r: r["total"], reverse=True)[:10]

        # Build vertical stacked bar chart
        apps = [r["application"] for r in rows]
        highs = [r["high"] for r in rows]
        meds = [r["medium"] for r in rows]
        lows = [r["low"] for r in rows]

        x = range(len(apps))

        fig, ax = plt.subplots(figsize=(max(6, len(apps) * 0.7), 4))

        # low at bottom, then medium, then high – all vertical
        low_bars = ax.bar(x, lows, label="Low", color="#28a745")      # green
        med_bars = ax.bar(x, meds, bottom=lows, label="Medium", color="#ffc107")  # yellow
        high_bottom = [l + m for l, m in zip(lows, meds)]
        high_bars = ax.bar(x, highs, bottom=high_bottom, label="High", color="#dc3545")  # red

        ax.set_xticks(list(x))
        ax.set_xticklabels(apps, rotation=45, ha="right")
        ax.set_ylabel("Number of Risks")
        ax.set_title("Application Risk Breakdown (High / Medium / Low)")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        chart_b64 = base64.b64encode(buf.read()).decode("utf-8")

        return {"rows": rows, "chart": chart_b64}

    # ----- Overall charts for criticality, risk item, status -----
    def _build_global_charts(
        self,
        df: pd.DataFrame,
        sem: Dict[str, Optional[str]]
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        charts: Dict[str, Optional[Dict[str, Any]]] = {
            "criticality": None,
            "risk_item": None,
            "status": None,
        }

        critical_col = sem.get("criticality") or sem.get("risk_level")
        if critical_col and critical_col in df.columns:
            charts["criticality"] = self.chart_agent.make_simple_chart(
                df, critical_col, preferred_type="pie", title="Application Criticality"
            )

        risk_item_col = sem.get("risk_item")
        if risk_item_col and risk_item_col in df.columns:
            charts["risk_item"] = self.chart_agent.make_simple_chart(
                df, risk_item_col, preferred_type="bar", title="Risk Item Distribution"
            )

        status_col = sem.get("status")
        if status_col and status_col in df.columns:
            # 👇 Status as PIE now
            charts["status"] = self.chart_agent.make_simple_chart(
                df, status_col, preferred_type="pie", title="Status Distribution"
            )

        return charts
