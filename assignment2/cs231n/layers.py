from builtins import range
from typing import Literal, LiteralString
import numpy as np


def affine_forward(x, w, b):
  """Computes the forward pass for an affine (fully connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)

  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  ###########################################################################
  # TODO: Copy over your solution from Assignment 1.                        #
  ###########################################################################
  # -1 means "let numpy infer this dimension automatically", we can also just write `np.prod(x.shape[1:])`
  flattened = np.reshape(x, shape=(x.shape[0], -1))
  out = flattened @ w + b
  ###########################################################################
  #                             END OF YOUR CODE                            #
  ###########################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """Computes the backward pass for an affine (fully connected) layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)
    - b: Biases, of shape (M,)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  ###########################################################################
  # TODO: Copy over your solution from Assignment 1.                        #
  ###########################################################################
  # NOTE: See my own notes (L3...)!!! Otherwise can't understand what it is about here!

  # corresponding to notes: Gi = G{i+1} @ Wi.T (no σ' since affine has no activation)
  dx_flat = dout @ w.T  # (N, D)
  dx = dx_flat.reshape(x.shape)

  # dWi = Zi.T @ G{i+1}
  dw = x.reshape(x.shape[0], -1).T @ dout  # (D, M)

  # bias gradient (not explicitly in notes)
  # dout = XW + b, chain rule: dl/db = dl/d(dout) * (d(dout) / db) = dl/d(dout) * 1
  #       and modify it considering the shape.
  db = np.sum(dout, axis=0)  # sum over batch
  ###########################################################################
  #                             END OF YOUR CODE                            #
  ###########################################################################
  return dx, dw, db


def relu_forward(x):
  """Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  ###########################################################################
  # TODO: Copy over your solution from Assignment 1.                        #
  ###########################################################################
  out = np.maximum(0, x)
  ###########################################################################
  #                             END OF YOUR CODE                            #
  ###########################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  ###########################################################################
  # TODO: Copy over your solution from Assignment 1.                        #
  ###########################################################################
  # dx[i] = dout[i] * 1  if x[i] > 0
  # dx[i] = dout[i] * 0  if x[i] <= 0
  # (x > 0) is just a shorthand of above
  dx = dout * (x > 0)
  ###########################################################################
  #                             END OF YOUR CODE                            #
  ###########################################################################
  return dx


def softmax_loss(x: np.ndarray, y: np.ndarray):
  """Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
    class for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  ###########################################################################
  # TODO: Copy over your solution from Assignment 1.                        #
  ###########################################################################
  exp_scores = np.exp(x - np.max(x, axis=1, keepdims=True))
  p = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
  loss: float = -np.sum(np.log(p[np.arange(x.shape[0]), y])) / x.shape[0]

  dx: np.ndarray = p
  dx[np.arange(x.shape[0]), y] -= 1
  dx /= x.shape[0]
  ###########################################################################
  #                             END OF YOUR CODE                            #
  ###########################################################################
  return loss, dx


def batchnorm_forward(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, bn_param: dict):
  """Forward pass for batch normalization.

  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the
  mean and variance of each feature, and these averages are used to normalize
  data at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7
  implementation of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode: str = bn_param["mode"]
  eps: float = bn_param.get("eps", 1e-5)
  momentum: float = bn_param.get("momentum", 0.9)

  running_mean: np.ndarray = bn_param.get(
    "running_mean", np.zeros(x.shape[1], dtype=x.dtype))
  running_var: np.ndarray = bn_param.get(
    "running_var", np.zeros(x.shape[1], dtype=x.dtype))

  out: np.ndarray
  cache: dict = {}

  if mode == "train":
    #######################################################################
    # TODO: Implement the training-time forward pass for batch norm.      #
    # Use minibatch statistics to compute the mean and variance, use      #
    # these statistics to normalize the incoming data, and scale and      #
    # shift the normalized data using gamma and beta.                     #
    #                                                                     #
    # You should store the output in the variable out. Any intermediates  #
    # that you need for the backward pass should be stored in the cache   #
    # variable.                                                           #
    #                                                                     #
    # You should also use your computed sample mean and variance together #
    # with the momentum variable to update the running mean and running   #
    # variance, storing your result in the running_mean and running_var   #
    # variables.                                                          #
    #                                                                     #
    # Note that though you should be keeping track of the running         #
    # variance, you should normalize the data based on the standard       #
    # deviation (square root of variance) instead!                        #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
    # might prove to be helpful.                                          #
    #######################################################################

    # i didnt add comment to explain it; it's all in the essay.
    # https://arxiv.org/pdf/1502.03167
    sample_mean: np.ndarray = np.mean(x, axis=0)
    sample_var: np.ndarray = np.var(x, axis=0)
    running_mean: np.ndarray = momentum * \
        running_mean + (1 - momentum) * sample_mean
    running_var: np.ndarray = momentum * \
        running_var + (1 - momentum) * sample_var

    cache['eps'] = eps
    cache['sample_var'] = sample_var
    cache['sample_mean'] = sample_mean

    std: np.ndarray = np.sqrt(sample_var + eps)
    x_hat: np.ndarray = (x - sample_mean) / std
    # ^^^^ better to use np.var and sqrt it rather than np.std, for numeric stability
    out: np.ndarray = gamma * x_hat + beta

    cache['std'] = std
    cache['x_hat'] = x_hat
    cache['x'] = x
    cache['gamma'] = gamma
    cache['beta'] = beta

    #######################################################################
    #                           END OF YOUR CODE                          #
    #######################################################################
  elif mode == "test":
    #######################################################################
    # TODO: Implement the test-time forward pass for batch normalization. #
    # Use the running mean and variance to normalize the incoming data,   #
    # then scale and shift the normalized data using gamma and beta.      #
    # Store the result in the out variable.                               #
    #######################################################################
    normalized_x: np.ndarray = (x - running_mean) / np.sqrt(running_var + eps)
    out: np.ndarray = gamma * normalized_x + beta
    #######################################################################
    #                          END OF YOUR CODE                           #
    #######################################################################
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param["running_mean"] = running_mean
  bn_param["running_var"] = running_var

  return out, cache


def batchnorm_backward(dout: np.ndarray, cache: dict):
  """Backward pass for batch normalization.

  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.

  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.

  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  ###########################################################################
  # TODO: Implement the backward pass for batch normalization. Store the    #
  # results in the dx, dgamma, and dbeta variables.                         #
  # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
  # might prove to be helpful.                                              #
  ###########################################################################
  # ditto, see original essay
  dgamma: np.ndarray = np.sum(cache['x_hat'] * dout, axis=0)
  dbeta: np.ndarray = np.sum(dout, axis=0)

  dx_hat: np.ndarray = dout * cache['gamma']
  dvar: np.ndarray = np.sum(
    dx_hat * (cache['x'] - cache['sample_mean']) * (-1 / 2) * np.pow(cache['std'], -3), axis=0)
  dmiu: np.ndarray = np.sum(dx_hat / (-cache['std']), axis=0) + \
      dvar / (cache['x'].shape[0]) * (-2) * \
      np.sum((cache['x'] - cache['sample_mean']), axis=0)

  dx: np.ndarray = dx_hat / cache['std'] + \
      dvar * 2 * (cache['x'] - cache['sample_mean']) / \
      (cache['x'].shape[0]) + dmiu / (cache['x'].shape[0])

  ###########################################################################
  #                             END OF YOUR CODE                            #
  ###########################################################################

  return dx, dgamma, dbeta


def batchnorm_backward_alt(dout: np.ndarray, cache: dict):
  """Alternative backward pass for batch normalization.

  For this implementation you should work out the derivatives for the batch
  normalizaton backward pass on paper and simplify as much as possible. You
  should be able to derive a simple expression for the backward pass.
  See the jupyter notebook for more hints.

  Note: This implementation should expect to receive the same cache variable
  as batchnorm_backward, but might not use all of the values in the cache.

  Inputs / outputs: Same as batchnorm_backward
  """
  ###########################################################################
  # TODO: Implement the backward pass for batch normalization. Store the    #
  # results in the dx, dgamma, and dbeta variables.                         #
  #                                                                         #
  # After computing the gradient with respect to the centered inputs, you   #
  # should be able to compute gradients with respect to the inputs in a     #
  # single statement; our implementation fits on a single 80-character line.#
  ###########################################################################
  # see essay
  N: int = dout.shape[0]
  D: int = dout.shape[1]

  dgamma: np.ndarray = np.sum(dout.T @ cache['x_hat'] * np.eye(D), axis=0)
  dbeta: np.ndarray = np.sum(dout, axis=0)
  dx_hat: np.ndarray = dout * cache['gamma']
  dvar: np.ndarray = np.sum(dx_hat * (cache['x'] - cache['sample_mean']) * (-1 / 2) * (
    cache['sample_var'] + cache['eps'])**(-3 / 2), axis=0)
  dmean: np.ndarray = np.sum(dx_hat * (-1) / (cache['sample_var'] + cache['eps'])**(
    1 / 2) + dvar * (-2) * (cache['x'] - cache['sample_mean']) / N, axis=0)
  dx: np.ndarray = dx_hat * (cache['sample_var'] + cache['eps'])**(-1 / 2) + \
      dvar * 2 * (cache['x'] - cache['sample_mean']) / N + dmean / N

  ###########################################################################
  #                             END OF YOUR CODE                            #
  ###########################################################################

  return dx, dgamma, dbeta


def layernorm_forward(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, ln_param: dict):
  """Forward pass for layer normalization.

  During both training and test-time, the incoming data is normalized per data-point,
  before being scaled by gamma and beta parameters identical to that of batch normalization.

  Note that in contrast to batch normalization, the behavior during train and test-time for
  layer normalization are identical, and we do not need to keep track of running averages
  of any sort.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - ln_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  eps: float = ln_param.get("eps", 1e-5)
  ###########################################################################
  # TODO: Implement the training-time forward pass for layer norm.          #
  # Normalize the incoming data, and scale and  shift the normalized data   #
  #  using gamma and beta.                                                  #
  # HINT: this can be done by slightly modifying your training-time         #
  # implementation of  batch normalization, and inserting a line or two of  #
  # well-placed code. In particular, can you think of any matrix            #
  # transformations you could perform, that would enable you to copy over   #
  # the batch norm code and leave it almost unchanged?                      #
  ###########################################################################
  # keepdims makes the shape as (N, 1) not (N, )
  # or use `.reshape(-1,1)`
  feature_mean: np.ndarray = np.mean(x, axis=1, keepdims=True)
  feature_var: np.ndarray = np.var(x, axis=1, keepdims=True)
  std: np.ndarray = np.sqrt(feature_var + eps)
  x_hat: np.ndarray = (x - feature_mean) / std
  out: np.ndarray = gamma * x_hat + beta

  cache: dict = {}
  cache['std'] = std
  cache['x_hat'] = x_hat
  cache['gamma'] = gamma
  cache['beta'] = beta
  cache['x'] = x
  cache['eps'] = eps
  cache['feature_mean'] = feature_mean
  cache['feature_var'] = feature_var
  ###########################################################################
  #                             END OF YOUR CODE                            #
  ###########################################################################
  return out, cache


def layernorm_backward(dout: np.ndarray, cache: dict):
  """Backward pass for layer normalization.

  For this implementation, you can heavily rely on the work you've done already
  for batch normalization.

  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from layernorm_forward.

  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  ###########################################################################
  # TODO: Implement the backward pass for layer norm.                       #
  #                                                                         #
  # HINT: this can be done by slightly modifying your training-time         #
  # implementation of batch normalization. The hints to the forward pass    #
  # still apply!                                                            #
  ###########################################################################
  dgamma: np.ndarray = np.sum(cache['x_hat'] * dout, axis=0)
  dbeta: np.ndarray = np.sum(dout, axis=0)

  dx_hat: np.ndarray = dout * cache['gamma']
  dvar: np.ndarray = np.sum(
    dx_hat * (cache['x'] - cache['feature_mean']) * (-1 / 2) * np.pow(cache['std'], -3), axis=1, keepdims=True)
  dmiu: np.ndarray = np.sum(dx_hat / (-cache['std']), axis=1, keepdims=True) + \
      dvar / (cache['x'].shape[1]) * (-2) * \
      np.sum((cache['x'] - cache['feature_mean']), axis=1, keepdims=True)

  dx: np.ndarray = dx_hat / cache['std'] + \
      dvar * 2 * (cache['x'] - cache['feature_mean']) / \
      (cache['x'].shape[1]) + dmiu / (cache['x'].shape[1])
  ###########################################################################
  #                             END OF YOUR CODE                            #
  ###########################################################################
  return dx, dgamma, dbeta


def dropout_forward(x: np.ndarray, dropout_param: dict):
  """Forward pass for inverted dropout.

  Note that this is different from the vanilla version of dropout.
  Here, p is the probability of keeping a neuron output, as opposed to
  the probability of dropping a neuron output.
  See http://cs231n.github.io/neural-networks-2/#reg for more details.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We keep each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not
      in real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p: float = dropout_param["p"]
  mode: LiteralString = dropout_param["mode"]

  if "seed" in dropout_param:
    np.random.seed(dropout_param["seed"])

  mask = None
  out: np.ndarray

  if mode == "train":
    #######################################################################
    # TODO: Implement training phase forward pass for inverted dropout.   #
    # Store the dropout mask in the mask variable.                        #
    #######################################################################
    # see https://cs231n.github.io/neural-networks-2/#reg
    mask = (np.random.rand(*x.shape) < p) / p
    out = x * mask
    #######################################################################
    #                           END OF YOUR CODE                          #
    #######################################################################
  elif mode == "test":
    #######################################################################
    # TODO: Implement the test phase forward pass for inverted dropout.   #
    #######################################################################
    out = x
    #######################################################################
    #                            END OF YOUR CODE                         #
    #######################################################################

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)  # type: ignore

  return out, cache


def dropout_backward(dout: np.ndarray, cache: tuple):
  """Backward pass for inverted dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param: dict
  mask: np.ndarray
  dropout_param, mask = cache
  mode = dropout_param["mode"]

  dx: np.ndarray

  if mode == "train":
    #######################################################################
    # TODO: Implement training phase backward pass for inverted dropout   #
    #######################################################################
    dx = dout * mask
    #######################################################################
    #                          END OF YOUR CODE                           #
    #######################################################################
  elif mode == "test":
    dx = dout
  return dx  # type: ignore


def get_x_shapes(x: np.ndarray) -> tuple[int, int, int, int]:
  """
  Helper function to extract shapes from input x. 
  USEFUL FOR SYNTAX HIGHLIGHTING.

  Returns N, C_IN, H, W.

  Inputs:
  - x: Input data of shape (N, C, H, W)

  Returns:
  - N: batch size
  - C_IN: number of channels
  - H: height
  - W: width
  """
  N: int = x.shape[0]
  C_IN: int = x.shape[1]
  H: int = x.shape[2]
  W: int = x.shape[3]
  return N, C_IN, H, W


def extract_conv_params(x: np.ndarray, w: np.ndarray, conv_param: dict)\
        -> tuple[int, int, int, int, int, int, int, int, int, int, int]:
  """
  Helper function to extract convolution parameters from inputs. 
  USEFUL FOR SYNTAX HIGHLIGHTING.

  Returns N, C_IN, H, W, C_OUT, KH, KW, H_OUT, W_OUT, PAD, STRIDE

  Inputs:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns:
  - N: batch size
  - C_IN: number of channels
  - H: height
  - W: width
  - C_OUT: number of filters
  - KH: filter height
  - KW: filter width
  - H_OUT: output height
  - W_OUT: output width
  - PAD: padding
  - STRIDE: stride
  """
  N, C_IN, H, W = get_x_shapes(x)
  C_OUT: int = w.shape[0]
  assert C_IN == w.shape[1]
  KH: int = w.shape[2]
  KW: int = w.shape[3]

  PAD: int = int(conv_param['pad'])
  STRIDE: int = int(conv_param['stride'])

  H_OUT = (H + 2 * PAD - KH) / STRIDE + 1
  W_OUT = (W + 2 * PAD - KW) / STRIDE + 1
  assert int(H_OUT) == H_OUT and int(W_OUT) == W_OUT

  H_OUT = int(H_OUT)
  W_OUT = int(W_OUT)

  return N, C_IN, H, W, C_OUT, KH, KW, H_OUT, W_OUT, PAD, STRIDE


def conv_forward_naive(x: np.ndarray, w: np.ndarray, b: np.ndarray, conv_param: dict):
  """A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and
  width W. We convolve each input with F different filters, where each filter
  spans all C channels and has height HH and width WW.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
  along the height and width axes of the input. Be careful not to modfiy the original
  input x directly.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  ###########################################################################
  # TODO: Implement the convolutional forward pass.                         #
  # Hint: you can use the function np.pad for padding.                      #
  ###########################################################################
  # based on my lecture notes(notation differs from the notes above)
  N, C_IN, H, W, C_OUT, KH, KW, H_OUT, W_OUT, PAD, STRIDE \
      = extract_conv_params(x, w, conv_param)
  out = np.zeros((N, C_OUT, int(H_OUT), int(W_OUT)), dtype=x.dtype)

  def i2ipad(n: int, c: int, i: int, j: int) -> float:
    if PAD <= i and i < H + PAD and PAD <= j and j < W + PAD:
      return x[n, c, i - PAD, j - PAD]
    else:
      return 0

  for n in range(N):
    for cout in range(C_OUT):
      for h in range(int(H_OUT)):
        for w_idx in range(int(W_OUT)):
          prod = .0
          for cin in range(C_IN):
            for kh in range(KH):
              for kw in range(KW):
                prod += i2ipad(n, cin, h * STRIDE + kh,
                               w_idx * STRIDE + kw) * w[cout, cin, kh, kw]
          out[n, cout, h, w_idx] = prod + b[cout]
  ###########################################################################
  #                             END OF YOUR CODE                            #
  ###########################################################################
  cache: tuple = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout: np.ndarray, cache: tuple):
  """A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  ###########################################################################
  # TODO: Implement the convolutional backward pass.                        #
  ###########################################################################
  x, w, b, conv_param = cache
  N, C_IN, H, W, C_OUT, KH, KW, H_OUT, W_OUT, PAD, STRIDE \
      = extract_conv_params(x, w, conv_param)

  db = np.zeros(b.shape)
  for cout in range(C_OUT):
    for n in range(N):
      for h in range(H_OUT):
        for w_idx in range(W_OUT):
          db[cout] += dout[n, cout, h, w_idx]

  def i2ipad(n: int, c: int, i: int, j: int) -> float:
    if PAD <= i and i < H + PAD and PAD <= j and j < W + PAD:
      return x[n, c, i - PAD, j - PAD]
    else:
      return 0

  dw = np.zeros(w.shape)
  for cout in range(C_OUT):
    for cin in range(C_IN):
      for kh in range(KH):
        for kw in range(KW):
          for n in range(N):
            for h in range(H_OUT):
              for w_idx in range(W_OUT):
                h_start = h * STRIDE
                w_start = w_idx * STRIDE
                dw[cout, cin, kh, kw] += i2ipad(n, cin, h_start +
                                                kh, w_start + kw) * dout[n, cout, h, w_idx]

  # compute padded dx, then unpad
  dx_padded = np.zeros((N, C_IN, H + 2 * PAD, W + 2 * PAD), dtype=x.dtype)

  for n in range(N):
    for cout in range(C_OUT):
      for h_out in range(H_OUT):
        for w_out in range(W_OUT):
          h_start = h_out * STRIDE
          w_start = w_out * STRIDE
          # w[cout] shape: (C_IN, KH, KW), broadcasting works!
          dx_padded[n, :, h_start:h_start + KH, w_start:w_start + KW] += \
              dout[n, cout, h_out, w_out] * w[cout]

  # dx shape (N, C_IN, H, W)
  dx = dx_padded[:, :, PAD:PAD + H, PAD:PAD + W]
  ###########################################################################
  #                             END OF YOUR CODE                            #
  ###########################################################################
  return dx, dw, db


def max_pool_forward_naive(x: np.ndarray, pool_param: dict):
  """A naive implementation of the forward pass for a max-pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  No padding is necessary here, eg you can assume:
    - (H - pool_height) % stride == 0
    - (W - pool_width) % stride == 0

  Returns a tuple of:
  - out: Output data, of shape (N, C, H', W') where H' and W' are given by
    H' = 1 + (H - pool_height) / stride
    W' = 1 + (W - pool_width) / stride
  - cache: (x, pool_param)
  """
  ###########################################################################
  # TODO: Implement the max-pooling forward pass                            #
  ###########################################################################
  N, C_IN, H, W = get_x_shapes(x)

  POOL_HEIGHT = int(pool_param['pool_height'])
  POOL_WIDTH = int(pool_param['pool_width'])
  STRIDE = int(pool_param['stride'])

  H_OUT = (H - POOL_HEIGHT) / STRIDE + 1
  W_OUT = (W - POOL_WIDTH) / STRIDE + 1

  assert int(H_OUT) == H_OUT and int(W_OUT) == W_OUT

  H_OUT = int(H_OUT)
  W_OUT = int(W_OUT)

  out: np.ndarray = np.zeros((N, C_IN, H_OUT, W_OUT))

  for n in range(N):
    for cin in range(C_IN):
      for hout in range(H_OUT):
        for wout in range(W_OUT):
          h_start = hout * STRIDE
          w_start = wout * STRIDE
          h_end = h_start + POOL_HEIGHT
          w_end = w_start + POOL_WIDTH
          out[n, cin, hout, wout] = np.max(
            x[n, cin, h_start:h_end, w_start:w_end])

  ###########################################################################
  #                             END OF YOUR CODE                            #
  ###########################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout: np.ndarray, cache: tuple):
  """A naive implementation of the backward pass for a max-pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  ###########################################################################
  # TODO: Implement the max-pooling backward pass                           #
  ###########################################################################
  x: np.ndarray
  pool_param: dict
  x, pool_param = cache

  N, C_IN, H, W = get_x_shapes(x)

  POOL_HEIGHT = int(pool_param['pool_height'])
  POOL_WIDTH = int(pool_param['pool_width'])
  STRIDE = int(pool_param['stride'])

  H_OUT = (H - POOL_HEIGHT) / STRIDE + 1
  W_OUT = (W - POOL_WIDTH) / STRIDE + 1

  assert int(H_OUT) == H_OUT and int(W_OUT) == W_OUT

  H_OUT = int(H_OUT)
  W_OUT = int(W_OUT)
  dx = np.zeros((N, C_IN, H, W))

  for n in range(N):
    for cin in range(C_IN):
      for hout in range(H_OUT):
        for wout in range(W_OUT):
          h_start = hout * STRIDE
          w_start = wout * STRIDE
          h_end = h_start + POOL_HEIGHT
          w_end = w_start + POOL_WIDTH
          pooling_window = x[n, cin, h_start:h_end, w_start:w_end]
          mask = (pooling_window == np.max(pooling_window))
          # `+=` may not be necessary if not overlapping,
          # also if multiple max, both are considered valid and get non-zero derivatives.
          dx[n, cin, h_start:h_end, w_start:w_end] += dout[n, cin, hout, wout] * mask


###########################################################################
#                             END OF YOUR CODE                            #
###########################################################################
  return dx


def spatial_batchnorm_forward(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, bn_param: dict):
  """Computes the forward pass for spatial batch normalization.

  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """

  ###########################################################################
  # TODO: Implement the forward pass for spatial batch normalization.       #
  #                                                                         #
  # HINT: You can implement spatial batch normalization by calling the      #
  # vanilla version of batch normalization you implemented above.           #
  # Your implementation should be very short; ours is less than five lines. #
  ###########################################################################
  # batchnorm_forward function expects input shape (N_batch, D_features)
  N_SAMPLE, C, H, W = get_x_shapes(x)
  # D_features is C_IN (channel in) here (normalization C unchanegd, C == C_IN == C_OUT!)
  reshaped_x = x.transpose(0, 2, 3, 1).reshape(-1, C)
  raw_out, cache = batchnorm_forward(reshaped_x, gamma, beta, bn_param)
  out = raw_out.reshape(N_SAMPLE, H, W, C).transpose(0, 3, 1, 2)
  ###########################################################################
  #                             END OF YOUR CODE                            #
  ###########################################################################

  return out, cache


def spatial_batchnorm_backward(dout: np.ndarray, cache: dict):
  """Computes the backward pass for spatial batch normalization.

  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass

  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """

  ###########################################################################
  # TODO: Implement the backward pass for spatial batch normalization.      #
  #                                                                         #
  # HINT: You can implement spatial batch normalization by calling the      #
  # vanilla version of batch normalization you implemented above.           #
  # Your implementation should be very short; ours is less than five lines. #
  ###########################################################################
  N_SAMPLE, C, H, W = get_x_shapes(dout)
  reshaped_dout = dout.transpose(0, 2, 3, 1).reshape(-1, C)
  raw_dx, dgamma, dbeta = batchnorm_backward(
    reshaped_dout, cache)
  dx = raw_dx.reshape(N_SAMPLE, H, W, C).transpose(0, 3, 1, 2)

  ###########################################################################
  #                             END OF YOUR CODE                            #
  ###########################################################################

  return dx, dgamma, dbeta


def spatial_groupnorm_forward(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, G: int, gn_param: dict):
  """Computes the forward pass for spatial group normalization.

  In contrast to layer normalization, group normalization splits each entry in the data into G
  contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
  are then applied to the data, in a manner identical to that of batch normalization and layer
  normalization.

  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (1, C, 1, 1)
  - beta: Shift parameter, of shape (1, C, 1, 1)
  - G: Integer mumber of groups to split into, should be a divisor of C
  - gn_param: Dictionary with the following keys:
    - eps: Constant for numeric stability

  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  eps: float = gn_param.get("eps", 1e-5)
  ###########################################################################
  # TODO: Implement the forward pass for spatial group normalization.       #
  # This will be extremely similar to the layer norm implementation.        #
  # In particular, think about how you could transform the matrix so that   #
  # the bulk of the code is similar to both train-time batch normalization  #
  # and layer normalization!                                                #
  ###########################################################################
  N, C, H, W = get_x_shapes(x)
  GROUP_SIZE = C / G
  assert GROUP_SIZE == int(GROUP_SIZE)
  GROUP_SIZE = int(GROUP_SIZE)
  layers = x.reshape(N, G, GROUP_SIZE, H, W)
  sample_means: np.ndarray = np.mean(layers, axis=2, keepdims=True)
  sample_vars: np.ndarray = np.var(layers, axis=2, keepdims=True)
  sample_stds: np.ndarray = np.sqrt(sample_vars + eps)
  raw_x_hat: np.ndarray = (layers - sample_means) / sample_stds
  x_hat: np.ndarray = raw_x_hat.reshape((N, C, H, W))
  out: np.ndarray = gamma * x_hat + beta

  cache = {}
  cache['eps'] = eps
  cache['sample_vars'] = sample_vars
  cache['sample_means'] = sample_means
  cache['std'] = sample_stds
  cache['x_hat'] = x_hat
  cache['gamma'] = gamma
  cache['beta'] = beta
  cache['layers'] = layers
  cache['x'] = x
  cache['G'] = G
  ###########################################################################
  #                             END OF YOUR CODE                            #
  ###########################################################################
  return out, cache


def spatial_groupnorm_backward(dout: np.ndarray, cache: dict):
  """Computes the backward pass for spatial group normalization.

  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass

  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
  - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
  """

  ###########################################################################
  # TODO: Implement the backward pass for spatial group normalization.      #
  # This will be extremely similar to the layer norm implementation.        #
  ###########################################################################
  dgamma: np.ndarray = np.sum(
    cache['x_hat'] * dout, axis=(0, 2, 3), keepdims=True)
  dbeta: np.ndarray = np.sum(dout, axis=(0, 2, 3), keepdims=True)

  N, C, H, W = get_x_shapes(dout)
  G: int = cache['G']
  GROUP_SIZE = C / G
  assert GROUP_SIZE == int(GROUP_SIZE)
  GROUP_SIZE = int(GROUP_SIZE)

  layers: np.ndarray = cache['layers']
  assert layers.shape == (N, G, GROUP_SIZE, H, W)

  draw_x_hat: np.ndarray = dout * cache['gamma']
  dx_hat: np.ndarray = draw_x_hat.reshape((N, G, GROUP_SIZE, H, W))
  dvar: np.ndarray = np.sum(
    dx_hat * (layers - cache['sample_means']) * (-1 / 2) * np.pow(cache['std'], -3), axis=2, keepdims=True)
  dmiu: np.ndarray = np.sum(dx_hat / (-cache['std']), axis=2, keepdims=True) + \
      dvar / (GROUP_SIZE) * (-2) * \
      np.sum((layers - cache['sample_means']), axis=2, keepdims=True)
  dlayers: np.ndarray = dx_hat / cache['std'] + \
      dvar * 2 * (layers - cache['sample_means']) / \
      (GROUP_SIZE) + dmiu / (GROUP_SIZE)
  dx: np.ndarray = dlayers.reshape((N, C, H, W))
  ###########################################################################
  #                             END OF YOUR CODE                            #
  ###########################################################################
  return dx, dgamma, dbeta
