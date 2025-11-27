import numpy as np

W_xh = np.array([[1], [0], [0]])
W_hh = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1]])
W_yh = np.array([[1, 1, -1]])
h_start = np.array([[0], [0], [1]])
X_SEQ = [0, 1, 0, 1, 1, 1, 0, 1, 1]


def relu(x: np.ndarray) -> np.ndarray:
  return np.maximum(x, 0)


h_states = [h_start]
outputs = []

for t, x in enumerate(X_SEQ):
  h_prev = h_states[t]
  h_t = relu(W_hh @ h_prev + W_xh * x)
  h_states.append(h_t)
  y_t = relu(W_yh @ h_t)
  outputs.append(y_t)

for xi, yi in zip(X_SEQ, outputs):
  print(xi, yi.reshape(-1))
