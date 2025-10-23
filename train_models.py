# train_models.py
# EDA, preprocessing, feature extraction (TF-IDF), train classifiers and priority model
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
import re

nltk.download('stopwords')
STOP = set(stopwords.words('english'))

os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# 1. Load data
df = pd.read_csv("dataset/tasks_synthetic.csv")
print("Rows:", len(df))
# Quick EDA (counts)
print(df['type'].value_counts())
print(df['priority'].value_counts())

# 2. Simple text clean function
def clean_text(s):
    s = str(s).lower()
    s = re.sub(r'http\S+',' ', s)
    s = re.sub(r'[^a-z0-9\s]',' ', s)
    tokens = [w for w in s.split() if w not in STOP and len(w)>2]
    return " ".join(tokens)

df['text'] = (df['title'].fillna('') + " " + df['description'].fillna('') + " " + df['tag'].fillna(''))
df['text_clean'] = df['text'].apply(clean_text)

# 3. Train/test split for classification (task type)
X = df['text_clean']
y = df['type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. TF-IDF vectorizer + classifier pipeline (compare NB & SVM)
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))

# Multinomial NB pipeline
nb_pipe = Pipeline([('tfidf', tfidf), ('clf', MultinomialNB())])
nb_pipe.fit(X_train, y_train)
y_pred_nb = nb_pipe.predict(X_test)
print("Naive Bayes classification report:\n", classification_report(y_test, y_pred_nb))

# SVM pipeline
svm_pipe = Pipeline([('tfidf', tfidf), ('clf', LinearSVC(random_state=42))])
# small GridSearch for SVM C
param_grid = {'clf__C': [0.1, 1, 5]}
svm_search = GridSearchCV(svm_pipe, param_grid, cv=3, n_jobs=-1, verbose=1)
svm_search.fit(X_train, y_train)
print("Best SVM params:", svm_search.best_params_)
y_pred_svm = svm_search.predict(X_test)
print("SVM classification report:\n", classification_report(y_test, y_pred_svm))

# Save best classifier (choose SVM if better)
# Compare f1-macro
from sklearn.metrics import f1_score
f1_nb = f1_score(y_test, y_pred_nb, average='macro')
f1_svm = f1_score(y_test, y_pred_svm, average='macro')
best_clf = svm_search.best_estimator_ if f1_svm >= f1_nb else nb_pipe
joblib.dump(best_clf, "models/classifier.pkl")
joblib.dump(tfidf, "models/tfidf.pkl")
print("Saved classifier and tfidf.")

# Confusion matrix plot for best classifier
cm = confusion_matrix(y_test, best_clf.predict(X_test), labels=best_clf.classes_)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=best_clf.classes_, yticklabels=best_clf.classes_)
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.title("Confusion Matrix - Task Type")
plt.tight_layout()
plt.savefig("plots/confusion_task_type.png")
plt.close()

# 5. Priority prediction (Low/Medium/High)
# Use features: text_clean + estimated_hours + days_left + tag (simple)
df2 = df.copy()
# convert tag to simple categorical via get_dummies
X_pr = pd.DataFrame()
X_pr['estimated_hours'] = df2['estimated_hours']
X_pr['days_left'] = df2['days_left'].fillna(0)
# TF-IDF transform text into numeric features (sparse)
tfidf_obj = tfidf.fit(df2['text_clean'])
text_feat = tfidf_obj.transform(df2['text_clean'])
# We'll use top 100 TF-IDF features to keep model simple
from sklearn.feature_selection import SelectKBest, chi2
skb = SelectKBest(chi2, k=100)
skb.fit(text_feat, df2['priority'])
text_sel = skb.transform(text_feat)
# combine numeric + text_sel
from scipy.sparse import hstack
X_num = df2[['estimated_hours','days_left']].fillna(0).values
X_comb = hstack([text_sel, X_num])

y_pr = df2['priority']
Xtr, Xte, ytr, yte = train_test_split(X_comb, y_pr, test_size=0.2, random_state=42, stratify=y_pr)

# Random Forest
rf = RandomForestClassifier(random_state=42, n_jobs=-1)
param_grid_rf = {'n_estimators':[100,200], 'max_depth':[5,10,None]}
rf_search = GridSearchCV(rf, param_grid_rf, cv=3, n_jobs=-1, verbose=1)
rf_search.fit(Xtr, ytr)
print("Best RF params:", rf_search.best_params_)
ypred_rf = rf_search.predict(Xte)
print("Priority classification report:\n", classification_report(yte, ypred_rf))

joblib.dump(rf_search.best_estimator_, "models/priority_model.pkl")
joblib.dump(skb, "models/sel_kbest.pkl")
joblib.dump(tfidf_obj, "models/tfidf_full.pkl")
print("Saved priority model and selection objects.")

# 6. Save small metadata
meta = {
    "classes_task": list(best_clf.classes_),
    "priority_classes": list(rf_search.best_estimator_.classes_) if hasattr(rf_search.best_estimator_, "classes_") else ['Low','Medium','High']
}
joblib.dump(meta, "models/meta.pkl")
print("Training finished. Models in models/*.pkl")
