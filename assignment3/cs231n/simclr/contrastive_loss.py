import torch
import numpy as np


def sim(z_i: torch.Tensor, z_j: torch.Tensor):
  """Normalized dot product between two vectors.

  Inputs:
  - z_i: 1xD tensor.
  - z_j: 1xD tensor.

  Returns:
  - A scalar value that is the normalized dot product between z_i and z_j.
  """
  ##############################################################################
  # TODO: Start of your code.                                                  #
  #                                                                            #
  # HINT: torch.linalg.norm might be helpful.                                  #
  ##############################################################################
  norm_dot_product: torch.Tensor = \
      (z_i @ z_j.T) / (torch.linalg.norm(z_i) * torch.linalg.norm(z_j))
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return norm_dot_product


def simclr_loss_naive(out_left: torch.Tensor, out_right: torch.Tensor, tau: float):
  """Compute the contrastive loss L over a batch (naive loop version).

  Input:
  - out_left: NxD tensor; output of the projection head g(), left branch in SimCLR model.
  - out_right: NxD tensor; output of the projection head g(), right branch in SimCLR model.
  Each row is a z-vector for an augmented sample in the batch. The same row in out_left and out_right form a positive pair.
  In other words, (out_left[k], out_right[k]) form a positive pair for all k=0...N-1.
  - tau: scalar value, temperature parameter that determines how fast the exponential increases.

  Returns:
  - A scalar value; the total loss across all positive pairs in the batch. See notebook for definition.
  """
  N, D = out_left.shape
  # Concatenate out_left and out_right into a 2*N x D tensor.
  out: torch.Tensor = torch.cat([out_left, out_right], dim=0)  # [2*N, D]

  total_loss: float = 0
  for k in range(N):  # loop through each positive pair (k, k+N)
    z_k, z_k_N = out[k], out[k + N]
    ##############################################################################
    # TODO: Start of your code.                                                  #
    #                                                                            #
    # Hint: Compute l(k, k+N) and l(k+N, k).                                     #
    ##############################################################################
    lkkn_dnom, lknk_dnom = .0, .0
    for i in range(2 * N):
      lkkn_dnom += torch.exp(sim(z_k, out[i]) / tau) if i != k else .0
      lknk_dnom += torch.exp(sim(z_k_N, out[i]) / tau) if i != k + N else .0
    lkkn = - torch.log(torch.exp(sim(z_k, z_k_N) / tau) / lkkn_dnom)
    lknk = - torch.log(torch.exp(sim(z_k_N, z_k) / tau) / lknk_dnom)
    total_loss = total_loss + lkkn + lknk  # type: ignore
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

  # In the end, we need to divide the total loss by 2N, the number of samples in the batch.
  total_loss = total_loss / (2 * N)
  return total_loss


def sim_positive_pairs(out_left: torch.Tensor, out_right: torch.Tensor):
  """Normalized dot product between positive pairs.

  Inputs:
  - out_left: NxD tensor; output of the projection head g(), left branch in SimCLR model.
  - out_right: NxD tensor; output of the projection head g(), right branch in SimCLR model.
  Each row is a z-vector for an augmented sample in the batch.
  The same row in out_left and out_right form a positive pair.

  Returns:
  - A Nx1 tensor; each row k is the normalized dot product between out_left[k] and out_right[k].
  """
  ##############################################################################
  # TODO: Start of your code.                                                  #
  #                                                                            #
  # HINT: torch.linalg.norm might be helpful.                                  #
  ##############################################################################
  # IMPORTANT: if we do not use `keepdim=True`, the result would (N,) instead of
  # (N, 1), i.e., during the next process it would be right or wrong.
  # if we apply `keepdim=True` partially, the shape would be (N, N) in this case
  # during broadcasting --> error occurs, however the program maybe runs!
  pos_pairs = torch.sum(out_left * out_right, dim=1, keepdim=True) / \
      (torch.linalg.norm(out_left, dim=1, keepdim=True) *
       torch.linalg.norm(out_right, dim=1, keepdim=True))
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return pos_pairs


def compute_sim_matrix(out: torch.Tensor) -> torch.Tensor:
  """Compute a 2N x 2N matrix of normalized dot products between all pairs of augmented examples in a batch.

  Inputs:
  - out: 2N x D tensor; each row is the z-vector (output of projection head) of a single augmented example.
  There are a total of 2N augmented examples in the batch.

  Returns:
  - sim_matrix: 2N x 2N tensor; each element i, j in the matrix is the normalized dot product between out[i] and out[j].
  """
  ##############################################################################
  # TODO: Start of your code.                                                  #
  ##############################################################################
  normalized = out / torch.linalg.norm(out, dim=1, keepdim=True)
  sim_matrix = normalized @ normalized.T
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return sim_matrix


def simclr_loss_vectorized(out_left: torch.Tensor, out_right: torch.Tensor, tau: float, device: torch.device | str = 'cuda'):
  """Compute the contrastive loss L over a batch (vectorized version). No loops are allowed.

  Inputs and output are the same as in simclr_loss_naive.
  """
  N = out_left.shape[0]

  # Concatenate out_left and out_right into a 2*N x D tensor.
  out: torch.Tensor = torch.cat([out_left, out_right], dim=0)  # [2*N, D]

  # Compute similarity matrix between all pairs of augmented examples in the batch.
  sim_matrix: torch.Tensor = compute_sim_matrix(out)  # [2*N, 2*N]

  ##############################################################################
  # TODO: Start of your code. Follow the hints.                                #
  ##############################################################################

  # Step 1: Use sim_matrix to compute the denominator value for all augmented samples.
  # Hint: Compute e^{sim / tau} and store into exponential, which should have shape 2N x 2N.
  exponential: torch.Tensor = torch.exp(sim_matrix / tau)  # (2N, 2N)

  # This binary mask zeros out terms where k=i.
  mask = (torch.ones_like(exponential, device=device) -
          torch.eye(2 * N, device=device)).to(device).bool()

  # We apply the binary mask.
  exponential = exponential.masked_select(mask).view(2 * N, -1)  # [2*N, 2*N-1]

  # Hint: Compute the denominator values for all augmented samples. This should be a 2N x 1 vector.
  denom: torch.Tensor = torch.sum(exponential, dim=1, keepdim=True)

  # Step 2: Compute similarity between positive pairs.
  # You can do this in two ways:
  # Option 1: Extract the corresponding indices from sim_matrix.
  # Option 2: Use sim_positive_pairs().
  # (2N, 1)
  similarities: torch.Tensor = \
      torch.cat([sim_positive_pairs(out_left, out_right),
                sim_positive_pairs(out_right, out_left)], dim=0)

  # Step 3: Compute the numerator value for all augmented samples.
  numerator: torch.Tensor = torch.exp(similarities / tau)

  # Step 4: Now that you have the numerator and denominator for all augmented samples, compute the total loss.
  loss = torch.sum(-torch.log(numerator / denom))
  loss /= 2 * N
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return loss


def rel_error(x, y):
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
