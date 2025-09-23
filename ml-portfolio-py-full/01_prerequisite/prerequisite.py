import numpy as np

# 벡터/행렬 기본
a = np.array([1,2,3], dtype=float)
b = np.array([4,5,6], dtype=float)
print("dot:", np.dot(a,b))

A = np.array([[1,2],[3,4]], dtype=float)
B = np.array([[5,6],[7,8]], dtype=float)
print("A @ B =\n", A @ B)

# 브로드캐스팅
x = np.arange(6).reshape(2,3)
print("broadcast add:\n", x + np.array([10,20,30]))

# 확률/통계
rng = np.random.default_rng(42)
samples = rng.normal(loc=0.0, scale=1.0, size=10000)
print("mean, var:", samples.mean(), samples.var())

# 선형회귀의 정규방정식 예시
X = np.c_[np.ones(5), np.arange(5)]
y = np.array([1,2,3,4,5.2])
theta = np.linalg.pinv(X) @ y
print("theta (closed-form):", theta)
