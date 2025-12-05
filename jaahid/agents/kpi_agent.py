# agents/kpi_agent.py

from typing import Dict, Any, Optional

import pandas as pd


class KPIAgent:
    """
    Risk-focused KPIs:
    - total risk items
    - high / medium / low
    - critical risks
    """

    def compute_risk_kpis(self, df: pd.DataFrame, sem: Dict[str, Optional[str]]) -> Dict[str, Any]:
        risk_level_col = sem.get("risk_level") or sem.get("criticality")
        total = len(df)

        high = medium = low = critical = 0

        if risk_level_col and risk_level_col in df.columns:
            series = df[risk_level_col]
            high, medium, low, critical = self._classify_levels(series)

        return {
            "total_risks": int(total),
            "high_risks": int(high),
            "medium_risks": int(medium),
            "low_risks": int(low),
            "critical_risks": int(critical),
        }

    def _classify_levels(self, series: pd.Series):
        """
        Try multiple strategies:
        - Textual: High/Medium/Low/Critical etc.
        - Short codes: H/M/L, Sev1/2/3, P1/P2/P3
        - Numeric: 1/2/3 or similar, or scores split by quantiles
        """
        s_str = series.astype(str).str.strip().str.lower()

        # --- textual high/medium/low ---
        high = s_str.str.contains(r"\bhigh\b").sum()
        medium = s_str.str.contains(r"\bmedium\b|\bmed\b").sum()
        low = s_str.str.contains(r"\blow\b").sum()
        critical = s_str.str.contains(r"\bcrit\b|\bcritical\b|\bsev1\b|\bp1\b").sum()

        if high + medium + low + critical > 0:
            if critical > 0 and critical > high:
                high = critical
            return high, medium, low, critical

        # --- short codes like H/M/L, 1/2/3, etc. ---
        high_codes = ["h", "3", "p1", "sev1"]
        med_codes = ["m", "2", "p2", "sev2"]
        low_codes = ["l", "1", "p3", "sev3"]

        high = s_str.isin(high_codes).sum()
        medium = s_str.isin(med_codes).sum()
        low = s_str.isin(low_codes).sum()
        critical = s_str.isin(["crit", "critical", "p1", "sev1"]).sum()

        if high + medium + low + critical > 0:
            if critical > 0 and critical > high:
                high = critical
            return high, medium, low, critical

        # --- numeric scores: use quantiles ---
        s_num = pd.to_numeric(series, errors="coerce")
        s_num = s_num.dropna()
        if len(s_num) == 0:
            return 0, 0, 0, 0

        uniq = sorted(s_num.unique())
        if 2 <= len(uniq) <= 6:
            low_val = uniq[0]
            high_val = uniq[-1]
            mid_idx = len(uniq) // 2
            med_val = uniq[mid_idx]

            low = (s_num == low_val).sum()
            medium = (s_num == med_val).sum()
            high = (s_num == high_val).sum()
            critical = 0
            return int(high), int(medium), int(low), int(critical)

        q1 = s_num.quantile(0.33)
        q2 = s_num.quantile(0.66)

        low = (s_num <= q1).sum()
        medium = ((s_num > q1) & (s_num <= q2)).sum()
        high = (s_num > q2).sum()
        critical = 0

        return int(high), int(medium), int(low), int(critical)
