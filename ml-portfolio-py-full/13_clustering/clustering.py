from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

X, _ = make_blobs(n_samples=500, centers=4, cluster_std=0.60, random_state=42)
kmeans = KMeans(n_clusters=4, n_init="auto", random_state=42).fit(X)
labels_k = kmeans.labels_
print("KMeans inertia:", kmeans.inertia_, "  silhouette:", silhouette_score(X, labels_k))

X2, _ = make_moons(n_samples=600, noise=0.08, random_state=42)
db = DBSCAN(eps=0.2, min_samples=5).fit(X2)
labels_d = db.labels_
print("DBSCAN clusters (including noise=-1):", set(labels_d))
