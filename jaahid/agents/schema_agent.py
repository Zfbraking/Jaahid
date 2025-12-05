# agents/schema_agent.py

import json
from typing import Optional, Dict

import pandas as pd

from llm_setup import llm  # your existing LLM wrapper

# 👉 EDIT THIS for your actual Excel headers (exact text as in row 1)
MANUAL_SCHEMA = {
    # examples – replace with your real column names from TCS_to_IT_Domain_v4.xlsx
    # "risk_level": "Risk Level",
    # "criticality": "Criticality",
    # "risk_desc": "Risk Description",
    # "owner": "Owner",
    # "due_date": "Due Date",
    # "application": "Application Name",
    # "status": "Status",
    # "risk_item": "Risk Item",
}
# Leave keys as-is; only change the values.


class SchemaAgent:
    """
    Agent for understanding structure with a focus on:
    - Grouping column (e.g., Workstream / Tower / Domain / Project)
    - Risk- and criticality-related columns:
      risk level, criticality, description, owner, due date
    - Application / Status / Risk item columns for visuals.
    """

    def __init__(self, llm_fn=llm):
        self.llm_fn = llm_fn

    # -------- Grouping column detection --------
    def detect_group_column(self, df: pd.DataFrame) -> Optional[str]:
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

        # Heuristic fallback if no LLM
        if not self.llm_fn:
            preferred = [
                "workstream", "tower", "domain", "area", "segment",
                "project", "program", "application", "app", "group"
            ]
            for col in candidates:
                cname = col.lower()
                if any(k in cname for k in preferred):
                    return col
            return candidates[0]

        prompt = f"""
You are analyzing a transition/risk Excel.

We want ONE column that is the best high-level grouping dimension
for the risk dashboard – similar to "Workstream", "Tower", "Domain",
"Application Group", "Project", or "Business Area".

Candidate columns (low unique values):
{json.dumps(metadata, indent=2)}

Return ONLY the name of the column (a single JSON string).
Example:
"Workstream"

Response MUST be pure JSON string, e.g. "Tower"
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

    # -------- Risk & criticality semantics + application/status/risk_item --------
    def detect_risk_semantics(self, df: pd.DataFrame) -> Dict[str, Optional[str]]:
        """
        Detect:
        - risk_level: risk level / severity / RAG / priority / numeric score
        - criticality: business/technical criticality or impact
        - risk_desc: risk/issue description
        - owner: responsible person
        - due_date: due/target/closure date
        - application: application/system/service column
        - status: item status column
        - risk_item: risk category/type column
        """

        # 0) Manual overrides first (most reliable)
        sem: Dict[str, Optional[str]] = {
            key: None for key in [
                "risk_level", "criticality", "risk_desc", "owner",
                "due_date", "application", "status", "risk_item"
            ]
        }
        for key, col_name in MANUAL_SCHEMA.items():
            if col_name and col_name in df.columns:
                sem[key] = col_name

        # If user provided everything we need, return immediately
        if sem["risk_level"] and sem["application"] and sem["status"]:
            # fill criticality with risk_level if still None
            if sem["criticality"] is None:
                sem["criticality"] = sem["risk_level"]
            return sem

        # 1) Build metadata for LLM / heuristics
        metadata = []
        for col in df.columns:
            nunique = df[col].nunique(dropna=True)
            metadata.append({
                "name": col,
                "dtype": str(df[col].dtype),
                "unique_values": int(nunique),
                "sample_values": [str(v) for v in df[col].dropna().unique()[:5]]
            })

        # 2) Heuristics – only fill what manual config didn't provide
        for col in df.columns:
            cname = col.lower()
            nunique = df[col].nunique(dropna=True)

            # risk level / severity / criticality
            if sem["risk_level"] is None and any(k in cname for k in [
                "risk level", "risk_level", "risk_rating", "risk rating", "risk",
                "severity", "impact", "criticality", "rag", "priority", "score"
            ]):
                if nunique <= 50:
                    sem["risk_level"] = col

            # description / issue text
            if sem["risk_desc"] is None and any(k in cname for k in [
                "risk desc", "risk_desc", "risk description", "issue",
                "description", "summary"
            ]):
                sem["risk_desc"] = col

            # owner
            if sem["owner"] is None and any(k in cname for k in [
                "owner", "assignee", "responsible", "owner name"
            ]):
                sem["owner"] = col

            # due / target date
            if sem["due_date"] is None and any(k in cname for k in [
                "due", "target date", "closure date", "end date", "finish"
            ]):
                sem["due_date"] = col

            # application / system
            if sem["application"] is None and any(k in cname for k in [
                "application", "app name", "app_name", "system", "service"
            ]):
                sem["application"] = col

            # status
            if sem["status"] is None and "status" in cname:
                sem["status"] = col

            # risk item / category / type
            if sem["risk_item"] is None and any(k in cname for k in [
                "risk item", "risk_item", "risk category", "risk type", "risktype", "riskcategory"
            ]):
                sem["risk_item"] = col

        if sem["criticality"] is None:
            sem["criticality"] = sem["risk_level"]

        # 3) If no LLM configured, just return what we have
        if not self.llm_fn:
            return sem

        # 4) LLM refinement – ask once for all semantics, then merge
        prompt = f"""
You are analyzing an Excel file that contains RISK and CRITICALITY information related to an IT transition.

We want to map the following concepts to actual column names:

- risk_level: ordinal risk level (High/Medium/Low, Critical, Severity, RAG, numeric score, etc.)
- criticality: business/technical criticality or impact/priority (can be same as risk_level)
- risk_desc: free text describing the risk/issue
- owner: person responsible / risk owner
- due_date: due/target/closure date of the risk
- application: application/system/service name being impacted
- status: current status of the item (Open, Closed, In Progress, etc.)
- risk_item: category/type of risk (e.g. Data, Infra, App, Process etc.)

Here is column metadata:
{json.dumps(metadata, indent=2)}

Return a JSON object like:
{{
  "risk_level": "<column name or null>",
  "criticality": "<column name or null>",
  "risk_desc": "<column name or null>",
  "owner": "<column name or null>",
  "due_date": "<column name or null>",
  "application": "<column name or null>",
  "status": "<column name or null>",
  "risk_item": "<column name or null>"
}}

Column names must match the metadata exactly. Use null if you are not sure.
Response MUST be pure JSON.
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

            for key in sem.keys():
                if sem[key] is None and valid(obj.get(key)):
                    sem[key] = obj[key]
        except Exception:
            pass

        if sem["criticality"] is None:
            sem["criticality"] = sem["risk_level"]

        return sem
