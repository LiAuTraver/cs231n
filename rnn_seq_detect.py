import numpy as np

# input to hidden matrix
# for this W_xh: current input value only directly influences the first element of the hidden state
W_xh = np.array([
  [1],
  [0],
  [0],
])
# hidden to hidden matrix
W_hh = np.array([
  [0, 0, 0],
  [1, 0, 0],
  [0, 0, 1],
])
# hidden to output
W_yh = np.array([
  [1, 1, -1],
])

# baseline mem
h_start = np.array([
  [0],
  [0],
  [1],
])


def relu(x: np.ndarray) -> np.ndarray:
  # y_t may be -1 if both the current and previous inputs are 0, and here set it to 0.
  return np.maximum(x, 0)


h_states = [h_start]
outputs: list[np.ndarray] = []

X_SEQ = [0, 1, 0, 1, 1, 1, 0, 1, 1]
for t, x in enumerate(X_SEQ):
  h_prev = h_states[t]
  h_t = relu(W_hh @ h_prev + W_xh * x)
  h_states.append(h_t)
  y_t = relu(W_yh @ h_t)
  outputs.append(y_t)

for xi, yi in zip(X_SEQ, outputs):
  # the output is 1 if and only if the current input is 1 and the previous input is also 1.(the y_1 is calculated as if the previous input is 0).
  print(xi, yi.reshape(-1))
