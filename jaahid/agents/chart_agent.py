# agents/chart_agent.py

from typing import Optional, Dict

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import base64


class ChartAgent:
    """
    Agent that generates charts for risk distributions.
    """

    # --- Grouped chart (per Workstream/Tower etc.) ---
    def make_chart_for_column(self, df_group: pd.DataFrame, col: str, group_value: str) -> Optional[Dict]:
        series = df_group[col]
        counts = series.value_counts(dropna=False)
        if counts.empty:
            return None

        counts.index = counts.index.map(lambda x: "NaN" if pd.isna(x) else str(x))

        cname = col.lower()
        # risk-like columns -> pie, fallback -> bar
        if any(k in cname for k in ["risk", "rag", "severity", "criticality", "impact", "priority"]):
            chart_type = "pie"
        else:
            chart_type = "bar"

        title = f"{group_value} – {col}"

        fig, ax = plt.subplots()
        if chart_type == "pie":
            counts.plot(kind="pie", ax=ax, autopct="%1.1f%%")
            ax.set_ylabel("")
        else:
            counts.plot(kind="bar", ax=ax)
            ax.set_ylabel("Count")
            ax.set_xlabel(col)

        ax.set_title(title)
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

    # --- Simple overall chart (no grouping) ---
    def make_simple_chart(
        self,
        df: pd.DataFrame,
        col: str,
        preferred_type: Optional[str] = None,
        title: Optional[str] = None
    ) -> Optional[Dict]:
        if col not in df.columns:
            return None

        series = df[col]
        counts = series.value_counts(dropna=False)
        if counts.empty:
            return None

        counts.index = counts.index.map(lambda x: "NaN" if pd.isna(x) else str(x))

        cname = col.lower()

        if preferred_type in ("pie", "bar", "line"):
            chart_type = preferred_type
        else:
            # auto pick
            if any(k in cname for k in ["risk", "rag", "severity", "criticality", "impact", "priority"]):
                chart_type = "pie"
            elif any(k in cname for k in ["status", "state"]):
                chart_type = "bar"
            else:
                chart_type = "bar"

        if title is None:
            title = col

        fig, ax = plt.subplots()
        if chart_type == "pie":
            counts.plot(kind="pie", ax=ax, autopct="%1.1f%%")
            ax.set_ylabel("")
        elif chart_type == "line":
            counts.sort_index().plot(kind="line", marker="o", ax=ax)
            ax.set_ylabel("Count")
            ax.set_xlabel(col)
            ax.grid(True, linestyle="--", alpha=0.5)
        else:
            counts.plot(kind="bar", ax=ax)
            ax.set_ylabel("Count")
            ax.set_xlabel(col)

        ax.set_title(title)
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
