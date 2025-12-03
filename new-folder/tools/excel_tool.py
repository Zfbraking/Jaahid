import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-GUI backend for Flask/server use
import matplotlib.pyplot as plt
import io, base64


def build_excel_tool(llm=None):
    class ExcelTool:
        name = "excel_visualizer"
        description = "Generate bar, pie, and line charts from tabular Excel/CSV data"

        def invoke(self, payload):
            file_path = payload.get("file_path")
            if not file_path:
                return {"error": "file_path is required"}

            try:
                lower = file_path.lower()
                if lower.endswith((".csv", ".txt")):
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path)
            except Exception as e:
                return {"error": f"Failed to read file as Excel/CSV: {e}"}

            # Suggestions mode: return only columns with duplicates
            if payload.get("suggestions_only"):
                categorical = []
                for c in df.columns:
                    # treat as categorical if object or low unique count
                    if df[c].dtype == "object" or df[c].nunique(dropna=True) < 30:
                        # only include columns that actually have duplicates
                        if df[c].duplicated().any():
                            categorical.append(c)

                return {"available_columns": ["All"] + categorical if categorical else []}

            # Chart mode
            x_col = payload.get("x_column")
            if not x_col or (x_col not in df.columns and x_col != "All"):
                return {"error": "Please select a valid column."}

            charts = []

            if x_col == "All":
                # generate charts only for categorical columns with duplicates
                for col in df.columns:
                    if (
                        df[col].dtype == "object"
                        or df[col].nunique(dropna=True) < 30
                    ) and df[col].duplicated().any():
                        charts.append(self._make_charts(df, col))
            else:
                # single selected column – also respect duplicates rule
                if not df[x_col].duplicated().any():
                    return {"error": f"Column '{x_col}' has all unique values; select a column with duplicates."}
                charts.append(self._make_charts(df, x_col))

            return {"charts": charts}

        def _make_charts(self, df, col):
            # Common series: counts of each category (including NaN)
            counts = df[col].value_counts(dropna=False)

            # ---------- Bar chart ----------
            fig_bar, ax_bar = plt.subplots()
            counts.plot(kind="bar", ax=ax_bar)
            ax_bar.set_xlabel(col)
            ax_bar.set_ylabel("Count")
            ax_bar.set_title(f"Count of {col}")
            fig_bar.tight_layout()

            buf_bar = io.BytesIO()
            fig_bar.savefig(buf_bar, format="png")
            plt.close(fig_bar)
            buf_bar.seek(0)
            bar_b64 = base64.b64encode(buf_bar.read()).decode("utf-8")

            # ---------- Pie chart ----------
            fig_pie, ax_pie = plt.subplots()
            counts.plot(kind="pie", ax=ax_pie, autopct="%1.1f%%")
            ax_pie.set_ylabel("")
            ax_pie.set_title(f"Distribution of {col}")
            fig_pie.tight_layout()

            buf_pie = io.BytesIO()
            fig_pie.savefig(buf_pie, format="png")
            plt.close(fig_pie)
            buf_pie.seek(0)
            pie_b64 = base64.b64encode(buf_pie.read()).decode("utf-8")

            # ---------- Line chart ----------
            # sorted by category label for a stable x-axis
            line_series = counts.sort_index()

            fig_line, ax_line = plt.subplots()
            line_series.plot(kind="line", marker="o", ax=ax_line)
            ax_line.set_xlabel(col)
            ax_line.set_ylabel("Count")
            ax_line.set_title(f"Line chart of {col} counts")
            ax_line.grid(True, linestyle="--", alpha=0.5)
            fig_line.tight_layout()

            buf_line = io.BytesIO()
            fig_line.savefig(buf_line, format="png")
            plt.close(fig_line)
            buf_line.seek(0)
            line_b64 = base64.b64encode(buf_line.read()).decode("utf-8")

            return {
                "column": col,
                "bar_chart": bar_b64,
                "pie_chart": pie_b64,
                "line_chart": line_b64,
            }

    return ExcelTool()
