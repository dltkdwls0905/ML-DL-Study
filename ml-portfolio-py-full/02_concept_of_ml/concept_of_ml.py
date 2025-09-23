from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

sc = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

clf = LogisticRegression(max_iter=1000).fit(Xtr_s, ytr)
pred = clf.predict(Xte_s)

print("Accuracy:", accuracy_score(yte, pred))
print("F1:", f1_score(yte, pred))
print(classification_report(yte, pred))
