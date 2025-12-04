import gradio as gr
import pandas as pd
import numpy as np
import random
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# -------------------------------
# Setup
# -------------------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# -------------------------------
# Generate synthetic datasets
# -------------------------------
def generate_datasets():
    # Classification
    X_cls, y_cls = make_classification(
        n_samples=2000, n_features=20, n_informative=6,
        n_redundant=4, n_classes=2, weights=[0.7,0.3],
        class_sep=1.2, flip_y=0.02, random_state=SEED
    )
    cls_cols = [f"feat_{i}" for i in range(X_cls.shape[1])]
    df_cls = pd.DataFrame(X_cls, columns=cls_cols)
    df_cls["target"] = y_cls
    df_cls["region"] = np.random.choice(["APAC","EMEA","AMER"], size=len(df_cls))
    df_cls["channel"] = np.random.choice(["web","mobile","api"], size=len(df_cls))

    # Regression
    X_reg, y_reg = make_regression(
        n_samples=1500, n_features=10, n_informative=5,
        noise=10.0, random_state=SEED
    )
    reg_cols = [f"rfeat_{i}" for i in range(X_reg.shape[1])]
    df_reg = pd.DataFrame(X_reg, columns=reg_cols)
    df_reg["target"] = y_reg
    df_reg["segment"] = np.random.choice(["A","B","C"], size=len(df_reg))

    return df_cls, df_reg

df_cls, df_reg = generate_datasets()

# -------------------------------
# Train models
# -------------------------------
def classification_metrics():
    df = pd.get_dummies(df_cls, columns=["region","channel"], drop_first=True)
    X = df.drop(columns=["target"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,
                                                        random_state=SEED,
                                                        stratify=y)
    rf = RandomForestClassifier(n_estimators=200, random_state=SEED)
    rf.fit(X_train,y_train)
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test,y_pred)
    return f"RandomForest Accuracy: {acc:.3f}"

def regression_metrics():
    df = pd.get_dummies(df_reg, columns=["segment"], drop_first=True)
    X = df.drop(columns=["target"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,
                                                        random_state=SEED)
    rf = RandomForestRegressor(n_estimators=200, random_state=SEED)
    rf.fit(X_train,y_train)
    y_pred = rf.predict(X_test)
    rmse = mean_squared_error(y_test,y_pred)**0.5
    return f"RandomForest RMSE: {rmse:.2f}"

# -------------------------------
# Incident analysis dataset
# -------------------------------
incident_data = {
    "IncidentID": [f"INC-{i}" for i in range(1, 11)],
    "ErrorType": ["DB","API","Config","Network","DB","API","Config","Network","DB","API"],
    "Downtime": [30,45,20,60,25,50,15,70,40,55],
    "Resolution": ["Rollback","Config Fix","Scaling","Failover",
                   "Rollback","Config Fix","Scaling","Failover",
                   "Rollback","Config Fix"]
}
df_inc = pd.DataFrame(incident_data)

# -------------------------------
# Incident analysis plots
# -------------------------------
def incident_analysis():
    fig1 = px.scatter(df_inc, x="ErrorType", y="Downtime", color="Resolution",
                      hover_data=["IncidentID","Resolution","Downtime"],
                      title="Incident Similarity (ErrorType vs Downtime)")

    res_counts = df_inc["Resolution"].value_counts().reset_index()
    res_counts.columns = ["Resolution", "Count"]
    fig2 = px.bar(res_counts, x="Resolution", y="Count", text="Count",
                  title="Resolution Frequency")

    fig3 = go.Figure(go.Sankey(
        node=dict(label=["Incident","Similar Cases","Rollback","Config Fix","Scaling","Failover","Recommendation"]),
        link=dict(source=[0,0,1,1,1,1], target=[1,2,3,4,5,6], value=[10,4,3,2,1,5])
    ))
    fig3.update_layout(title_text="Flow of incident analysis", font_size=12)

    return fig1, fig2, fig3

# -------------------------------
# Dynamic Recommendations
# -------------------------------
def recommend_remediation(incident_id):
    row = df_inc[df_inc["IncidentID"] == incident_id].iloc[0]

    # Rule-based recommendations
    if row["ErrorType"] == "DB":
        rec = "Rollback to last stable DB snapshot and verify schema consistency."
    elif row["ErrorType"] == "API":
        rec = "Apply configuration fixes, add API gateway monitoring."
    elif row["ErrorType"] == "Config":
        rec = "Review deployment configs, enforce automated validation."
    elif row["ErrorType"] == "Network":
        rec = "Trigger failover to backup network path, add redundancy."
    else:
        rec = "General remediation: investigate logs and apply rollback if needed."

    # Downtime adjustment
    if row["Downtime"] > 50:
        rec += " Since downtime is high, prioritize scaling or failover."

    return f"Incident {incident_id} ({row['ErrorType']} error, {row['Downtime']} mins): {rec}"

# -------------------------------
# Build Gradio interface
# -------------------------------
with gr.Blocks() as demo:
    gr.Markdown("## 🚀 Hackathon ML Demo (Gradio)")

    with gr.Tab("Classification"):
        gr.Dataframe(df_cls.head(), label="Classification Dataset Preview")
        gr.Textbox(value=classification_metrics(), label="Model Accuracy")

    with gr.Tab("Regression"):
        gr.Dataframe(df_reg.head(), label="Regression Dataset Preview")
        gr.Textbox(value=regression_metrics(), label="Model RMSE")

    with gr.Tab("Incident Analysis"):
        f1, f2, f3 = incident_analysis()
        gr.Plot(f1)
        gr.Plot(f2)
        gr.Plot(f3)

    with gr.Tab("Recommendations & Remediation"):
        incident_selector = gr.Dropdown(choices=df_inc["IncidentID"].tolist(),
                                        label="Select Incident")
        output_box = gr.Textbox(label="Recommendation", interactive=False)

        incident_selector.change(fn=recommend_remediation,
                                 inputs=incident_selector,
                                 outputs=output_box)

demo.launch()