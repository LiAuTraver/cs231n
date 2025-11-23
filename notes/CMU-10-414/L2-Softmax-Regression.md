# Lecture 2: Softmax Regression

## Three _ingredients_ of a machine learning algorithm

1. The hypothesis class: The _program structure_, parameterized via a set of parameters, that describes how we map inputs to outputs.
2. The loss function
3. An optimization method

## Loss function

Assuming

$$
\begin{aligned}
\mathbf{x} & \in \mathbb{R}^n \text{  (input feature vector)} \\
y & \in \{1, 2, \ldots, k\} \text{  (class label)} \\
\mathbf{h}(\mathbf{x}) & = [h_1(\mathbf{x}), h_2(\mathbf{x}), \ldots, h_k(\mathbf{x})]^T \in \mathbb{R}^k \text{  (hypothesis output vector)}
\end{aligned}
$$

> _for more clarity, each $h_i$ is a scalar-valued function of $\mathbf{x}$ hence $\mathbf{h}$ is a vector-valued function whose input is a vector._

### Classification loss functions

Simplest loss function:

$$
l_{\text{err}}(h(\mathbf{x}), y) =
\begin{cases}
  0 & \text{if argmax}_i \ \  h_i(\mathbf{x}) = y \\
  1 & \text{otherwise}
\end{cases}
$$

Although it looks simple, unfortunately, the error is a bad loss function for _optimization_, for it's not differentiable.

### Softmax/Cross-entropy loss

Let's convert that to a _probabilistic_ model by exponentiating and normalizing its entries

$$
z_i = p(\text{label } = i | \mathbf{x}) = \frac{e^{h_i(\mathbf{x})}}{\sum_{j=1}^k e^{h_j(\mathbf{x})}} \Leftrightarrow \mathbf{z} \equiv \text{normalize}(e^{\mathbf{h}(\mathbf{x})}) = \text{softmax}(\mathbf{h}(\mathbf{x}))
$$

> Note: The exponential function is applied element-wise. For example, if $\mathbf{h}(\mathbf{x}) = [1, 2, 3]^T$, then $e^{\mathbf{h}(\mathbf{x})} = [e^1, e^2, e^3]^T$. And in almost all cases in the Machine Learning context, when we write $e^{\mathbf{non-scalar}}$, we mean the **element-wise** application of the exponential function, not the matrix/vector exponential.

Then we can define the loss function as the negative log-probability of the correct class

$$
l_{\text{cross-entropy}}(\mathbf{h}(\mathbf{x}), y) = -\log(z_y) = -\log\left(\frac{e^{h_y(\mathbf{x})}}{\sum_{j=1}^k e^{h_j(\mathbf{x})}}\right) = -h_y(\mathbf{x}) + \log\left(\sum_{j=1}^k e^{h_j(\mathbf{x})}\right)
$$

### Regularization

Regularization is added in order to:

1. Express preference over weights
2. Avoid overfitting
3. Improve optimizationby adding curvature

$\lambda > 0$ is the regularization strength hyperparameter.

$$
\lambda R(\mathbf{W})
$$

L2 Regularization (Ridge):

$$
R(\mathbf{W}) = \|\mathbf{W}\|_F^2 = \sum_{i,j} W_{ij}^2
$$

L1 Regularization (Lasso):

$$
R(\mathbf{W}) = \|\mathbf{W}\|_1 = \sum_{i,j} |W_{ij}|
$$

Elastic Net (Combination of L1 and L2):

$$
R(\mathbf{W}) = \alpha \|\mathbf{W}\|_1 + (1 - \alpha) \|\mathbf{W}\|_F^2
$$

where $\alpha \in [0, 1]$ controls the balance between L1 and L2 regularization.

#### Regularizer Preference

L2 Regularization likes to _spread out_ weights, while L1 Regularization likes to _sparsify_ weights (make many weights exactly zero).

### Overall objective

$$
L(\mathbf{\theta}) = \underbrace{\frac{1}{m} \sum_{i=1}^m l_{\text{ce}}(\mathbf{h}_\theta(\mathbf{x}^{(i)}), y^{(i)})}_{\text{data loss(training)}} + \underbrace{\lambda R(\mathbf{\theta})}_{\text{regularization(avoid overfitting)}}
$$

### Sanity check

When $\mathbf{h}(\mathbf{x})$ roughly similar accross all classes, i.e., no class is particularly favored then it shall be uniform distribution after softmax, i.e., $z_i \approx \dfrac{1}{k}$.

Hence, if we got initial loss around $\log(k)$, then everything is fine.

When $\mathbf{h}(\mathbf{x})$ strongly favors the correct class $y$, i.e., $h_y(\mathbf{x}) \gg h_j(\mathbf{x}), \forall j \neq y$, then $z_y \approx 1$, hence the loss $l_{\text{ce}} \approx 0$. Everything is still fine.

### Optimization: Gradient Descent

How to optimize that scary loss function $l_{\text{ce}}$?

$$
\text{minimize}_{\theta} \ \dfrac{1}{m} \sum_{i=1}^m l_{\text{ce}}(\mathbf{h}_\theta(\mathbf{x}^{(i)}), y^{(i)})
$$

#### Quick recap

1. usually in order to avoid overflow/underflow when computing softmax, we can subtract the maximum value from all entries before exponentiating, i.e., for $\mathbf{h}(\mathbf{x})$, compute $\mathbf{h}(\mathbf{x}) - \max_i h_i(\mathbf{x})$ first, then apply softmax; the result is the same.

2. for a matrix-scalar function $f: \mathbb{R}^{n \times k} \to \mathbb{R}$'s gradient **Hessian matrix** is also a matrix of the same shape as its input, where each entry is the partial derivative of $f$ w.r.t. that entry of the input.

$$
\nabla_\mathbf{\theta} f(\mathbf{\theta}) \in \mathbb{R}^{n \times k} =
\begin{bmatrix}
  \dfrac{\partial f}{\partial \theta_{11}} & \dfrac{\partial f}{\partial \theta_{12}} & \ldots & \dfrac{\partial f}{\partial \theta_{1k}} \\ \\
  \dfrac{\partial f}{\partial \theta_{21}} & \dfrac{\partial f}{\partial \theta_{22}} & \ldots & \dfrac{\partial f}{\partial \theta_{2k}} \\
  \vdots & \vdots & \ddots & \vdots \\
  \dfrac{\partial f}{\partial \theta_{n1}} & \dfrac{\partial f}{\partial \theta_{n2}} & \ldots & \dfrac{\partial f}{\partial \theta_{nk}}
\end{bmatrix}
$$

...and points in the direction that most increases $f$ (locally). Using gradient descent, where $\alpha$ is the learning rate (step size)...

$$
\mathbf{\theta} := \mathbf{\theta} - \alpha \nabla_\mathbf{\theta}f(\mathbf{\theta})
$$

#### Stochastic Gradient Descent

Instead of computing the gradient over the entire dataset (batch gradient descent), we can compute the gradient based upon a minibatch(small partion of the data). For example, pick $B$ out of $m$ data points randomly, and compute the gradient based on those $B$ points, thus the update formula becomes

$$
\mathbf{\theta} := \mathbf{\theta} - \dfrac{\alpha}{B} \sum_{i = 1}^B \nabla_\mathbf{\theta} l_{\text{ce}}(\mathbf{h}_\theta(\mathbf{x}^{(i)}), y^{(i)})
$$

#### Anyway, we still have to compute the gradient for the softmax objective

$$
\nabla_\mathbf{\theta} l_{\text{ce}}(\mathbf{\theta}^T \mathbf{x}, y) = ?
$$

Actually it's very cumbersome to compute it correctly, we can assume $\mathbf{h}$ is an arbitrary vector rather than a linear function of $\mathbf{x}$, and compute the gradient w.r.t. $\mathbf{h}$ first:

$$
\dfrac{\partial l_{\text{ce}}}{\partial h_i} = ... = z_i - 1\{i = y\} \Rightarrow \nabla_\mathbf{h} l_{\text{ce}} = \mathbf{z} - \text{onehot}_k(y)
$$

> **Onehot**: a vector of length $k$ with all zeros except a one at the $y$-th position.

#### Hack

We _can_ use the _right way_ to compute the gradient using matrix differential calculus, Jacobians, Kronecker products, vectorization, etc. But it's too complicated and not very useful in practice. Hence there's a hacky way: _pretend everything possible as a scalar, (here it is $\mathbf{x} \to x$) and then rearrange/transpose them to make the sizes match with some numerical check._

> Why I wrote down the **_hack_** used here? Because that is what researchers actually use in practice.

##### Fake gradients

Pretend $x$ is a scalar, then

$$
\dfrac{\partial l_{\text{ce}}(\mathbf{\theta}^T x, y)}{\partial \mathbf{\theta}} = \dfrac{\partial l_{\text{ce}}}{\partial \mathbf{\theta}^T x} \dfrac{\partial \mathbf{\theta}^T x}{\partial \mathbf{\theta}} = (\mathbf{z} - \text{onehot}_k(y)) x
$$

##### Hence...

Analyze it: $z - \text{onehot}_k(y) \in \mathbb{R}^{k \times 1}$ with $\mathbf{x} \in \mathbb{R}^{n \times 1}$, hence the product is a $k \times n$ matrix, but we want an $n \times k$ matrix. So we just transpose it: and with both surprise and joy, it works!

$$
\nabla_\mathbf{\theta} l_{\text{ce}}(\mathbf{\theta}^T \mathbf{x}, y) = \mathbf{x} (\mathbf{z} - \text{onehot}_k(y))^T \in \mathbb{R}^{n \times k}
$$

Apply the similar hack for the batch form of the loss

$$
\nabla_{\mathbf{\theta}} l_{\text{ce}}(\mathbf{X}\mathbf{\theta},y) = \mathbf{X}^T (\mathbf{z} - \text{onehot}_k(y)) \in \mathbb{R}^{n \times k}
$$

to make it size-independent, we often divide the gradient by the batch size $m$.
