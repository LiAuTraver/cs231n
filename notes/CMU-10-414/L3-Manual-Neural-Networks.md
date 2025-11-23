# Lecture 3: "Manual" Neural Networks

## Nonlinear classification

One idea: apply a linear classifier to some (potentially) higher-dimensional features of the data
$$\mathbf{h}_\theta(\mathbf{x}) = \theta^T \phi(\mathbf{x})$$
$$\text{where  } \theta \in \mathbb{R}^{d \times k}, \phi: \mathbb{R}^n \to \mathbb{R}^d$$

> Note: $\theta$ here represents a linear transformation, and $\phi$ is not linear, if both were linear, it's obvious that $\phi$ and $\theta$ can be combined into a single linear function $\hat{\theta}$, and the whole thing would be meaningless, and the trained model would handle the complex relationships in reality poorly.

> Monotonic functions, like ReLU, Sigmoid aren't _linear_ - although it preserves order; it still breaks the linearity of the whole function, combined with the simplicity of those functions, they are widely used as the nonlinear activation functions in neural networks.

Essentially any nonlinear function of linear features
$$\phi(\mathbf{x}) = \sigma(\mathbf{W}^T \mathbf{x})$$
where $\mathbf{W} \in \mathbb{R}^{n \times d}$ and $\sigma: \mathbb{R^d} \to \mathbb{R^d}$ is a nonlinear function.

Example, let $\mathbf{W}$ be a fixed Gaussian random matrix, and $\sigma$ be a cosine function, then it is a **random Fourier feature map** which works pretty well in practice.

## Neural networks

> _Neural network_ is a cool name and borrowed from neuroscience, although the connection to real neural networks in brains is tenuous.

begin with the simplest form of neural network:

$$
\mathbf{h}_\theta(\mathbf{x}) = \mathbf{W_2}^T \sigma(\mathbf{W_1}^T \mathbf{x})
$$

$$
\theta = \{
  \mathbf{W_1} \in \mathbb{R}^{n \times d},
  \mathbf{W_2} \in \mathbb{R}^{d \times k}
\}
$$

again, $\sigma$ is a **nonlinear function applied element-wise**.

### Batch form

For $m$ data points

$$
\mathbf{X} =
\begin{bmatrix}
  -{\mathbf{x}^{(1)}}^T- \\
  -{\mathbf{x}^{(2)}}^T- \\
  \vdots \\
  -{\mathbf{x}^{(m)}}^T-
\end{bmatrix}
\in \mathbb{R}^{m \times n}
$$

Then the batch form of the above neural network is

$$
\mathbf{h}_\theta(\mathbf{X}) = \sigma(\mathbf{X} \mathbf{W_1}) \mathbf{W_2}
$$

which looks simpler and nicer; also it can be seen as a simple two-layer neural network.

### Gradients

As in Lesson 2, we can use the result and the hack to solve the gradient w.r.t. $\mathbf{W}_1$ and $\mathbf{W}_2$ respectively.

$$
\begin{aligned}
\nabla_{\mathbf{W}_1} l_{\mathrm{ce}}\big(\sigma(\mathbf{X}\mathbf{W}_1)\mathbf{W}_2,\,y\big)
&= \mathbf{X}^T \Big( \sigma'(\mathbf{X}\mathbf{W}_1) \odot \big( (\mathbf{z} - \operatorname{onehot}_k(y))\,\mathbf{W}_2^T \big) \Big)
\in \mathbb{R}^{n \times d} \\
\nabla_{\mathbf{W}_2} l_{\mathrm{ce}}\big(\sigma(\mathbf{X}\mathbf{W}_1)\mathbf{W}_2,\,y\big)
&= \sigma(\mathbf{X}\mathbf{W}_1)^T \big( \mathbf{z} - \operatorname{onehot}_k(y) \big)
\in \mathbb{R}^{d \times k}
\end{aligned}
$$

> $\odot$ represents the element-wise product (_Hadamard product_) between two matrices of the same shape.

### Backpropagation: Forward and backward passes

Consider

$$
\mathbf{Z}_{i + 1} = \sigma_i(\mathbf{Z}_i \mathbf{W}_i), \quad i = 1, 2, \ldots, L - 1
$$

#### Forward pass

Initialize: $\mathbf{Z}_1 = \mathbf{X}$

Iterate: $\mathbf{Z}_{i + 1} = \sigma_i(\mathbf{Z}_i \mathbf{W}_i), \quad i = 1, 2, \ldots, L - 1$

#### Backward pass

Initialize: $$\mathbf{G}_{L+1} = \nabla_{\mathbf{Z}_{L+1}} l(\mathbf{Z}_{L+1}, y) = \mathbf{z} - \operatorname{onehot}_k(y) $$

Iterate: $$\mathbf{G}_i =(\mathbf{G}_{i + 1} \odot \sigma'_i(\mathbf{Z}_i \mathbf{W}_i)) \mathbf{W}_i^T, \quad i = L, L - 1, \ldots, 1$$

#### All the gradients

$$
\nabla_{\mathbf{W}_i} l(\mathbf{Z}_{L + 1}, y) = \mathbf{Z}_i^T (\mathbf{G}_{i + 1} \odot \sigma'_i(\mathbf{Z}_i \mathbf{W}_i)) \in \mathbb{R}^{d_{i} \times d_{i +1 }}
$$

#### The Bias

Here we omitted bias for simplicity, but it is easy to add it back in practice by augmenting the input with an additional constant feature (usually set to 1).

<!-- TODO -->
