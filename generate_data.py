# generate_data.py
# Synthetic task dataset generator - realistic Trello/Jira style tasks
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

TITLES = [
    "Fix login error", "Add search feature", "Update user profile UI",
    "Database migration", "Write unit tests", "Optimize image loading",
    "Investigate crash on submit", "Create onboarding flow", "Refactor auth module",
    "Add export CSV"
]
DESCS = [
    "When user logs in, sometimes get 500 error. Steps to reproduce: ...",
    "Need to implement a search bar with fuzzy matching and filters.",
    "Profile page layout looks cramped on mobile devices. Improve spacing.",
    "Add migration scripts to move old schema to new one. Ensure backups.",
    "Add unit tests for payment module to improve coverage to 80%.",
    "Speed up image loading using lazy loading and compression.",
    "App crashes intermittently when submitting large form payloads.",
    "Design and implement onboarding wizard for new users.",
    "Refactor authentication flow to use OAuth2 best practices.",
    "Allow users to export orders in CSV and Excel format."
]
TAGS = ["backend","frontend","devops","testing","research","data","ux"]

ASSIGNEES = ["Alice","Bob","Chitra","Dev","Esha"]

def random_deadline(created, max_days=20):
    return created + timedelta(days=random.randint(1, max_days))

def priority_from_rule(days_left, est_hours, tag):
    # deterministic-ish mapping for synthetic label
    score = 0
    if days_left <= 2: score += 3
    elif days_left <= 5: score += 2
    else: score += 0
    if est_hours >= 8: score += 2
    if tag in ("backend","devops"): score += 1
    if score >=5: return "High"
    if score >=3: return "Medium"
    return "Low"

rows = []
for i in range(1000):  # dataset size
    created = datetime.now() - timedelta(days=random.randint(0,30))
    title = random.choice(TITLES)
    desc = random.choice(DESCS)
    tag = random.choice(TAGS)
    est_hours = random.choice([1,2,4,6,8,12])
    due = random_deadline(created, max_days=20)
    days_left = (due - datetime.now()).days
    priority = priority_from_rule(days_left, est_hours, tag)
    task_type = random.choices(["bug","feature","chore"], weights=[0.35,0.45,0.20])[0]
    assignee = random.choice(ASSIGNEES)
    rows.append({
        "id": f"TASK{i+1:04d}",
        "title": title,
        "description": desc,
        "tag": tag,
        "estimated_hours": est_hours,
        "created_at": created.strftime("%Y-%m-%d"),
        "due_date": due.strftime("%Y-%m-%d"),
        "days_left": (due - datetime.now()).days,
        "priority": priority,
        "type": task_type,
        "assignee": assignee
    })

df = pd.DataFrame(rows)
df.to_csv("dataset/tasks_synthetic.csv", index=False)
print("Saved dataset/tasks_synthetic.csv (rows = {})".format(len(df)))
