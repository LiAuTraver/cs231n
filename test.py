import numpy as np


dims = [4096] * 7

hs = []

x = np.random.randn(16, dims[0])

# test for forward pass
for di, do in zip(dims[:-1], dims[:-1]):
  # W = np.random.randn(di, do) * .05  # .1 will overflow .01 will vanish
  W = np.random.randn(di, do) * np.sqrt(2 / di)  # Kaiming He initialization
  x = np.maximum(0, x.dot(W))
  hs.append(x)


# plot each layer's mean and std
for i, h in enumerate(hs):
  print(f"Layer {i + 1}: mean={h.mean():.5f}, std={h.std():.5f}")
