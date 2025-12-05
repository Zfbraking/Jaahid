# agents/viz_planner_agent.py

import json
from typing import List

import pandas as pd

from llm_setup import llm


class VizPlannerAgent:
    """
    Agent that decides which RISK/CRITICALITY related columns to visualize as charts.
    """

    def __init__(self, llm_fn=llm):
        self.llm_fn = llm_fn

    def select_risk_visual_columns(self, df: pd.DataFrame, group_col: str) -> List[str]:
        candidates = []
        metadata = []

        for col in df.columns:
            if col == group_col:
                continue
            cname = col.lower()

            # only risk/criticality columns
            if not any(k in cname for k in [
                "risk", "severity", "critical", "criticality", "impact", "priority", "rag"
            ]):
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
            return candidates

        prompt = f"""
We are building a RISK & CRITICALITY dashboard.

We have chosen '{group_col}' as the grouping dimension.
Now we want to pick which RISK/CRITICALITY columns to show as charts.
Examples: risk level, severity, impact, criticality, risk RAG, risk priority.

Candidate columns (already filtered to risk-ish names):
{json.dumps(metadata, indent=2)}

Return ONLY a JSON array of column names to visualize, e.g.:
["Risk Level", "Severity"]

Your response MUST be pure JSON.
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

        return candidates
