from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

docs = [
    "genomic medicine with machine learning",
    "deep learning for precision medicine",
    "classical music and art history"
]

tfidf = TfidfVectorizer().fit_transform(docs)
sim = cosine_similarity(tfidf)
print("Cosine similarity matrix:\n", sim)
