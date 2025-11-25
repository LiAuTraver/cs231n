# Lecture 5: Convolutional Neural Networks

### Convolutional layers

- Input tensor: (Channels, Height, Width) = $(C_{in}, H, W)$, Batch of images:  
  $$\text{shape}(\mathbf{I}) = (N, C_{in}, H, W)$$

- Filter tensor: (Number of filters, Channels, Filter height, Filter width)  
  $$\text{shape}(\mathbf{F}) = (C_{out}, C_{in}, K_h, K_w)$$

- Bias tensor: (Number of filters,)
  $$\text{shape}(\mathbf{B}) = (C_{out},)$$
- Output tensor: (Number of filters, Output height, Output width)
  $$\text{shape}(\mathbf{O}) = (N, C_{out}, H_{out}, W_{out})$$
  where

  $$
  H_{out} = \frac{H + 2P - K_h}{S} + 1
  $$

  $$
  W_{out} = \frac{W + 2P - K_w}{S} + 1
  $$

  - $P$: padding
  - $S$: stride

- padding: addressing the problem of shrinking feature maps by adding zeros around the border of the input. Common padding $P = \frac{K - 1}{2}$ for odd-sized filters to maintain spatial dimensions.

- stride: as the network deepens, the receptive field increases. Using a stride $S > 1$ downsamples the input and further increases the receptive field exponentially(and may also reduce depths and computational cost).

### Strict mathematical definition

#### Forward pass

Let

$$
\begin{aligned}
\mathbf{I} & \in \mathbb{R}^{N \times C_{in} \times H \times W} \\
\mathbf{F} & \in \mathbb{R}^{C_{out} \times C_{in} \times K_h \times K_w} \\
\mathbf{B} & \in \mathbb{R}^{C_{out}} \\
P & \in \mathbb{Z}^+ \\
S & \in \mathbb{Z} \ \ \text{and}  \ \ S \geq 1 \\
\end{aligned}
$$

Define the $\mathbf{I}_{pad} \in \mathbb{R}^{N \times C_{in} \times (H + 2P) \times (W + 2P)}$ by

$$
\mathbf{I}_{pad}[n, c, i, j] =
\begin{cases}
\mathbf{I}[n, c, i - P, j - P] & \text{if } P \leq i < H + P \text{ and } P \leq j < W + P \\
0 & \text{otherwise}
\end{cases}
$$

where

$$
\begin{aligned}
n & \in \{0, \ldots, N - 1\} \\
c & \in \{0, \ldots, C_{in} - 1\} \\
i & \in \{0, \ldots, H + 2P - 1\} \\
j & \in \{0, \ldots, W + 2P - 1\} \\
\end{aligned}
$$

and satisfies

$$
\begin{aligned}
H_{out} & = \frac{H + 2P - K_h}{S} + 1 \in \mathbb{Z}^+ \\
W_{out} & = \frac{W + 2P - K_w}{S} + 1 \in \mathbb{Z}^+ \\
\end{aligned}
$$

set

$$
\begin{aligned}
h_{start} & = h \cdot S \\
h_{end} & = h_{start} + K_h \\
w_{start} & = w \cdot S \\
w_{end} & = w_{start} + K_w \\
\end{aligned}
$$

Then the convolution output is

$$

\boxed{
  \mathbf{O}[n, c_{out}, h, w] = \sum_{c_{in} = 0}^{C_{in} - 1} \sum_{k_h = 0}^{K_h - 1} \sum_{k_w = 0}^{K_w - 1} \mathbf{F}[c_{out}, c_{in}, k_h, k_w] \cdot \mathbf{I}_{pad}[n, c_{in}, h_{start} + k_h, w_{start} + k_w] + \mathbf{B}[c_{out}]
}
$$

> note: assumes zero-based indexing and padding value is 0, and bias is added per output channel and broadcast across spatial positions and batch.

#### Backward pass

Let the scalar loss be $L$. Let the upstream gradient be

$$
\mathbf{G} = \frac{\partial L}{\partial \mathbf{O}} \in \mathbb{R}^{N \times C_{\text{out}} \times H_{\text{out}} \times W_{\text{out}}}.
$$

We return gradients for bias $\mathbf{B}$, filters $\mathbf{F}$, and input $\mathbf{I}$. Keep $P$, $S$ and the padded input $\mathbf{I}_{pad}$ as defined above.

##### Gradient w.r.t. bias

Bias is added per output channel and broadcast over batch and spatial positions, therefore:

$$
\boxed{
\frac{\partial L}{\partial \mathbf{B}[c_{\text{out}}]} = \sum_{n=0}^{N-1} \sum_{h=0}^{H_{\text{out}}-1} \sum_{w=0}^{W_{\text{out}}-1} \mathbf{G}[n, c_{\text{out}}, h, w].
}
$$

with shape $\mathbb{R}^{C_{\text{out}}}$.

##### Gradient w.r.t. filters

Each filter element multiplies a corresponding window of the padded input in the forward pass, so

$$
\boxed{
\frac{\partial L}{\partial \mathbf{F}[c_{\text{out}}, c_{\text{in}}, k_h, k_w]} = \sum_{n=0}^{N-1} \sum_{h=0}^{H_{\text{out}}-1} \sum_{w=0}^{W_{\text{out}}-1} \mathbf{G}[n, c_{\text{out}}, h, w] \cdot \mathbf{I}_{pad}[n, c_{\text{in}}, h_{start} + k_h, w_{start} + k_w]
}
$$

with shape $\mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times K_h \times K_w}$.

##### Gradient w.r.t. padded input

A given element of $\mathbf{I}_{pad}[n, c_{in}, u, v]$ contributes to every output $\mathbf{O}[n, c_{out}, h, w]$ whose receptive field includes $(u, v)$. Equivalently:

$$
\boxed{
\frac{\partial L}{\partial \mathbf{I}_{pad}[n, c_{in}, u, v]} = \sum_{c_{out}=0}^{C_{out}-1} \sum_{h=0}^{H_{out}-1} \sum_{w=0}^{W_{out}-1} \mathbf{G}[n, c_{out}, h, w] \cdot \mathbf{F}[c_{out}, c_{in}, u - h_{start}, v - w_{start}] \cdot \mathbf{1}_{0 \leq u - h_{start} < K_h} \cdot \mathbf{1}_{0 \leq v - w_{start} < K_w}
}
$$

where $\mathbf{1}_{condition}$ is an indicator function that is 1 if the condition is true and 0 otherwise, with shape $\mathbb{R}^{N \times C_{in} \times (H + 2P) \times (W + 2P)}$.

##### Gradient w.r.t. input

After removing padding, the gradient on the original input is the central slice of the padded-input gradient.

$$
\boxed{
\frac{\partial L}{\partial \mathbf{I}[n, c_{in}, x, y]} = \frac{\partial L}{\partial \mathbf{I}_{pad}[n, c_{in}, x + P, y + P]}, \quad x \in \{0, \ldots, H-1\}, \; y \in \{0, \ldots, W-1\}.
}
$$

### Pooling - Another way to downsample the feature map

**Max Pooling**: for each non-overlapping $p \times p$ region, take the maximum value. (most common)

###### Stride?

for max pooling usually equals the pooling size to avoid overlapping regions.

###### Padding

usually no padding for max pooling, it's no use to add zeros around the border.

###### ReLU

Conbine max pooling with ReLU is redundant, since most often they both output non-negative values; max pooling is non-linear itself; however suppose we use average pooling, which is linear, then ReLU is still useful.

### Math - A simple pooling without padding

Let input be

$$
\mathbf{I} \in \mathbb{R}^{N \times C \times H \times W}
$$

Set hyperparameters:

- Kernel size(of pooling): $K$
- Stride: $S$
- Pooling function: $f: \mathbb{R}^{K \times K} \rightarrow \mathbb{R}$ (e.g., max, average)

> Note: no learnable parameters here.

The the size of output is

$$
\begin{aligned}
H_{out} & = \frac{H - K}{S} + 1 \\
W_{out} & = \frac{W - K}{S} + 1
\end{aligned}
$$

_assuming these are integers._

Output is

$$
\boxed{
\mathbf{O}[n, c, h, w] = f(\mathbf{I}[n, c, h_{start}:h_{end}, w_{start}:w_{end}])
}
$$
