from sklearn.datasets import make_moons
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

X, y = make_moons(n_samples=600, noise=0.25, random_state=42)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)

svm = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True).fit(Xtr, ytr)
pred = svm.predict(Xte)
print("Test Acc:", accuracy_score(yte, pred))
