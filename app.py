# app.py - Final AI Task Manager Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from scipy.sparse import hstack
import os

st.set_page_config(page_title="AI Task Manager (Demo)", layout="centered")
st.title("AI-Powered Task Management System (Demo)")

# Load models
clf = joblib.load("models/classifier.pkl")
tfidf = joblib.load("models/tfidf.pkl")
priority_model = joblib.load("models/priority_model.pkl")
sel_kbest = joblib.load("models/sel_kbest.pkl")
tfidf_full = joblib.load("models/tfidf_full.pkl")

# Sample user workload (hrs)
user_workload = {
    "Alice": 12,
    "Bob": 8,
    "Chitra": 20,
    "Dev": 5,
    "Esha": 3
}

st.sidebar.header("Team Workload (hrs)")
for u,h in user_workload.items():
    st.sidebar.write(f"{u}: {h} hrs")

# Workload bar chart
st.sidebar.bar_chart(pd.Series(user_workload))

st.header("Add a New Task")
title = st.text_input("Title", "Fix login error")
description = st.text_area("Description", "When user logs in, sometimes get 500 error...")
tag = st.selectbox("Tag", ["backend","frontend","devops","testing","ux","data"], index=0)
est_hours = st.number_input("Estimated hours", min_value=1, max_value=100, value=4)
due_date = st.date_input("Due date", datetime.now().date())

if st.button("Predict & Assign"):
    # Text preprocessing
    import nltk, re
    from nltk.corpus import stopwords
    nltk.download('stopwords', quiet=True)
    stop = set(stopwords.words('english'))

    text = f"{title} {description} {tag}".lower()
    s = re.sub(r'[^a-z0-9\s]',' ', text)
    tokens = [w for w in s.split() if w not in stop and len(w)>2]
    clean = " ".join(tokens)

    # Task type prediction
    pred_type = clf.predict([clean])[0]
    st.success(f"Predicted task type: {pred_type}")

    # Priority prediction
    text_vec = tfidf_full.transform([clean])
    text_sel = sel_kbest.transform(text_vec)
    num = np.array([[est_hours, (due_date - datetime.now().date()).days]])
    Xp = hstack([text_sel, num])
    pr = priority_model.predict(Xp)[0]
    st.info(f"Predicted priority: {pr}")

    # Assignment heuristic
    candidates = sorted(user_workload.items(), key=lambda x: x[1])
    assigned = None
    for u,h in candidates:
        if h + est_hours <= 40:
            assigned = u
            break
    if not assigned:
        assigned = candidates[0][0]

    st.write(f"Suggested assignee: **{assigned}** (current workload: {user_workload[assigned]} hrs)")

    # Save assignment to CSV
    os.makedirs("outputs", exist_ok=True)
    assign_df = pd.DataFrame([{
        "title": title,
        "description": description,
        "tag": tag,
        "estimated_hours": est_hours,
        "due_date": due_date,
        "predicted_type": pred_type,
        "predicted_priority": pr,
        "assigned_to": assigned
    }])
    assign_csv = "outputs/assignments.csv"
    if os.path.exists(assign_csv):
        assign_df.to_csv(assign_csv, mode='a', header=False, index=False)
    else:
        assign_df.to_csv(assign_csv, index=False)
    st.success(f"Assignment saved to `{assign_csv}`")

# Priority distribution chart
st.header("Priority Distribution (Demo)")
try:
    df_prior = pd.read_csv("dataset/tasks_synthetic.csv")
    st.bar_chart(df_prior['priority'].value_counts())
except:
    st.warning("Dataset not found for distribution chart.")

st.markdown("---")
st.markdown("### Next Steps")
st.markdown("""
- Save tasks to project board (DB/Trello API)  
- Notify assignee (email/Slack)  
- Rebalance if urgent or skill mismatch
""")
