# ======================================
# === EVENT RISK TOOL: FILE START    ===
# ======================================
"""
Event risk forecasting tool.

Tool name: risk_forecaster

Modes:
  - "index":
        payload: { "mode": "index", "file_path": "<path to csv/xlsx>" }
        Reads the log file, parses timestamps, and stores an in-memory DataFrame.

  - "forecast":
        payload: {
            "mode": "forecast",
            "dimension": "severity" | "service" | "error_code" | "message",
            "value": "<e.g. CRITICAL, payment, ERR_1279, 'Out of memory'>",
            "window": "day" | "week" | "month"
        }

        Returns:
          - expected_count (Poisson λ)
          - probability_at_least_one (1 - exp(-λ))
          - basic metadata
"""

from typing import Dict, Any
import math

import pandas as pd

from tools.mcp_tooling import MCPTool

# Global in-memory log DataFrame
LOG_DF = None


def build_risk_tool() -> MCPTool:
    def _run(payload: Dict[str, Any]) -> Dict[str, Any]:
        mode = (payload.get("mode") or "").lower()
        if mode == "index":
            return _index_logs(payload)
        elif mode == "forecast":
            return _forecast_risk(payload)
        else:
            return {"error": "Invalid mode. Use 'index' or 'forecast'."}

    return MCPTool(
        name="risk_forecaster",
        description="Forecast risk (expected count, probability) of future events like CRITICAL or specific service errors.",
        func=_run,
    )


def _read_log_file(file_path: str) -> pd.DataFrame:
    """Read CSV or Excel log file and parse timestamp."""
    lower = file_path.lower()
    if lower.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    # normalize timestamp column name
    if "timestamp" not in df.columns:
        # try a few common variants
        for cand in ["time", "datetime", "event_time"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "timestamp"})
                break

    if "timestamp" not in df.columns:
        raise ValueError("No 'timestamp' column found in log file.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    return df


def _index_logs(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Load the log file into memory."""
    global LOG_DF

    file_path = payload.get("file_path")
    if not file_path:
        return {"error": "file_path is required for indexing logs."}

    try:
        df = _read_log_file(file_path)
    except Exception as e:
        return {"error": f"Failed to read log file: {e}"}

    if df.empty:
        return {"error": "Log file appears to be empty after parsing timestamps."}

    # Ensure we have at least these columns where possible
    LOG_DF = df
    return {
        "status": "indexed",
        "row_count": len(df),
        "columns": list(df.columns),
    }


def _forecast_risk(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Forecast risk using a simple Poisson model based on historical daily rate.

    Steps:
      - Filter rows where df[dimension] == value
      - Aggregate daily counts over the full date range (including zero days)
      - Compute mean daily rate λ_day
      - For window:
          day   -> λ = λ_day
          week  -> λ = 7 * λ_day
          month -> λ = 30 * λ_day
      - Probability at least one event: p = 1 - exp(-λ)
    """
    global LOG_DF

    if LOG_DF is None:
        return {"error": "No log data indexed. Call mode='index' first."}

    dimension = payload.get("dimension")
    value = payload.get("value")
    window = (payload.get("window") or "day").lower()

    if not dimension or not value:
        return {"error": "Both 'dimension' and 'value' are required for forecasting."}

    if dimension not in LOG_DF.columns:
        return {"error": f"Column '{dimension}' not found in indexed log data."}

    if window not in ("day", "week", "month"):
        return {"error": "window must be one of: 'day', 'week', 'month'."}

    df = LOG_DF.copy()
    if "timestamp" not in df.columns:
        return {"error": "Indexed log data has no 'timestamp' column."}

    df["date"] = df["timestamp"].dt.date

    # Filter by dimension = value
    filtered = df[df[dimension] == value]
    if filtered.empty:
        return {
            "dimension": dimension,
            "value": value,
            "window": window,
            "expected_count": 0.0,
            "probability_at_least_one": 0.0,
            "detail": "No historical events found for this dimension/value.",
        }

    # Daily counts for matching events
    daily_counts = (
        filtered.groupby("date").size().sort_index()
    )  # Series indexed by date

    # Build full date range to account for days with zero events
    min_date = df["date"].min()
    max_date = df["date"].max()
    all_days = pd.date_range(min_date, max_date, freq="D")
    all_counts = daily_counts.reindex(all_days.date, fill_value=0)

    if len(all_counts) == 0:
        return {"error": "No valid daily data for forecasting."}

    lambda_daily = float(all_counts.mean())

    if window == "day":
        lam = lambda_daily
    elif window == "week":
        lam = lambda_daily * 7.0
    else:  # month
        lam = lambda_daily * 30.0

    # Poisson probability of at least one event: 1 - exp(-λ)
    prob_at_least_one = 1.0 - math.exp(-lam)

    return {
        "dimension": dimension,
        "value": value,
        "window": window,
        "lambda_daily": lambda_daily,
        "expected_count": lam,
        "probability_at_least_one": prob_at_least_one,
        "total_days_observed": len(all_counts),
        "total_events_for_value": int(all_counts.sum()),
    }

# ====================================
# === EVENT RISK TOOL: FILE END    ===
# ====================================
