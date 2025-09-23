from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X, y = load_iris(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(max_depth=3, random_state=42).fit(Xtr, ytr)
print(export_text(clf, feature_names=load_iris().feature_names))
print("Test Acc:", clf.score(Xte, yte))

# 트리 시각화 (이미지 파일 저장)
plt.figure(figsize=(10,6))
plot_tree(clf, feature_names=load_iris().feature_names, class_names=load_iris().target_names, filled=True)
plt.tight_layout(); plt.savefig("decision_tree_iris.png")
print("Saved: decision_tree_iris.png")
