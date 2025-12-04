# pyright: reportArgumentType=false
import torch
import tqdm
import math

import tqdm.asyncio

# in the essay the noise is denoted by epsilon


class GaussianDiffusion(torch.nn.Module):
  def __init__(
      self,
      model,
      *,
      image_size: int,
      timesteps: int = 1000,
      objective: str = "pred_noise",
      beta_schedule: str = "sigmoid",
  ):
    super().__init__()

    self.model = model
    self.channels: int = 3
    self.image_size: int = image_size
    self.objective: str = objective
    assert objective in {
        "pred_noise",
        "pred_x_start",
    }, "objective must be either pred_noise (predict noise) or pred_x_start (predict image start)"

    # A helper function to register some constants as buffers to ensure that
    # they are on the same device as model parameters.
    # See https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    # Each buffer can be accessed as `self.name`
    def register_buffer(name, val):
      return self.register_buffer(name, val.float())

    #############################################################################
    # Noise schedule beta and alpha values
    #############################################################################
    betas = get_beta_schedule(beta_schedule, timesteps)
    self.num_timesteps = int(betas.shape[0])
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)  # alpha_bar_t
    register_buffer("betas", betas)  # can be accessed as self.betas
    register_buffer("alphas", alphas)  # can be accessed as self.alphas
    register_buffer("alphas_cumprod", alphas_cumprod)  # self.alphas_cumprod

    #############################################################################
    # Other coefficients needed to transform between x_t, x_0, and noise
    # Note that according to Eq. (4) and its reparameterization in Eq. (14),
    # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
    # where noise is sampled from N(0, 1)
    #############################################################################
    register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
    register_buffer(
        "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
    )
    # register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
    # register_buffer(
    #     "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
    # )

    #############################################################################
    # For posterior q(x_{t-1} | x_t, x_0) according to Eq. (6) and (7) of the paper.
    #############################################################################
    # alpha_bar_{t-1}
    alphas_cumprod_prev = torch.nn.functional.pad(
      alphas_cumprod[:-1], (1, 0), value=1.0)
    register_buffer(
        "posterior_mean_coef1",
        betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
    )
    register_buffer(
        "posterior_mean_coef2",
        (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
    )
    posterior_var = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    posterior_std = torch.sqrt(posterior_var.clamp(min=1e-20))
    register_buffer("posterior_std", posterior_std)

    #################################################################
    # loss weight
    #################################################################
    snr: torch.Tensor = alphas_cumprod / (1 - alphas_cumprod)
    loss_weight = torch.ones_like(snr) if objective == "pred_noise" else snr
    register_buffer("loss_weight", loss_weight)

  def normalize(self, img):
    return img * 2 - 1

  def unnormalize(self, img):
    return (img + 1) * 0.5

  def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
    """Get x_start from x_t and noise according to Eq. (14) of the paper.
    Args:
        x_t: (b, *) tensor. Noisy image.
        t: (b,) tensor. Time step.
        noise: (b, *) tensor. Noise from N(0, 1).
    Returns:
        x_start: (b, *) tensor. Starting image.
    """
    ####################################################################
    # TODO:
    # Transform x_t and noise to get x_start according to Eq.(4) and Eq.(14).
    # Look at the coeffs in `__init__` method and use the `extract` function.
    ####################################################################
    # ^^^ comment was misleading, we only need Eq.(4) here
    x_start: torch.Tensor = (x_t - extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * noise) \
        / extract(self.sqrt_alphas_cumprod, t, x_t.shape)

    return x_start

  def predict_noise_from_start(self, x_t: torch.Tensor, t: torch.Tensor, x_start: torch.Tensor):
    """Get noise from x_t and x_start according to Eq. (14) of the paper.
    Args:
        x_t: (b, *) tensor. Noisy image.
        t: (b,) tensor. Time step.
        x_start: (b, *) tensor. Starting image.
    Returns:
        pred_noise: (b, *) tensor. Predicted noise.
    """
    ####################################################################
    # TODO:
    # Transform x_t and noise to get x_start according to Eq.(4) and Eq.(14).
    # Look at the coeffs in `__init__` method and use the `extract` function.
    ####################################################################
    # ^^^ misleading too, also `x_start` is passed in, not to solve for
    pred_noise: torch.Tensor = (x_t - extract(self.sqrt_alphas_cumprod, t, x_t.shape) *
                                x_start) / extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
    ####################################################################
    return pred_noise

  def q_posterior(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
    """Get the posterior q(x_{t-1} | x_t, x_0) according to Eq. (6) and (7) of the paper.
    Args:
        x_start: (b, *) tensor. Predicted start image.
        x_t: (b, *) tensor. Noisy image.
        t: (b,) tensor. Time step.
    Returns:
        posterior_mean: (b, *) tensor. Mean of the posterior.
        posterior_std: (b, *) tensor. Std of the posterior.
    """
    posterior_mean = None
    posterior_std = None
    ####################################################################
    # We have already implemented this method for you.
    c1 = extract(self.posterior_mean_coef1, t, x_t.shape)
    c2 = extract(self.posterior_mean_coef2, t, x_t.shape)
    posterior_mean = c1 * x_start + c2 * x_t
    posterior_std = extract(self.posterior_std, t, x_t.shape)
    ####################################################################
    return posterior_mean, posterior_std

  @torch.no_grad()
  def p_sample(self, x_t:torch.Tensor, t: int, model_kwargs:dict={}):
    """Sample from p(x_{t-1} | x_t) according to Eq. (6) of the paper. Used only during inference.
    Args:
        x_t: (b, *) tensor. Noisy image.
        t: int. Sampling time step.
        model_kwargs: additional arguments for the model.
    Returns:
        x_tm1: (b, *) tensor. Sampled image.
    """
    tf = torch.full((x_t.shape[0],), t,
                    device=x_t.device, dtype=torch.long)  # (b,)

    ##################################################################
    # TODO: Implement the sampling step p(x_{t-1} | x_t) according to Eq. (6):
    #
    # - Steps:
    #   1. Get the model prediction by calling self.model with appropriate args.
    #   2. The model output can be either noise or x_start depending on self.objective.
    #      You can recover the other by calling self.predict_start_from_noise or
    #      self.predict_noise_from_start as needed.
    #   3. Clamp predicted x_start to the valid range [-1, 1]. This ensures the
    #      generation remains stable during denoising iterations.
    #   4. Get the mean and std for q(x_{t-1} | x_t, x_0) using self.q_posterior,
    #      and sample x_{t-1}.
    ##################################################################
    out = self.model(x_t, tf, model_kwargs)

    if self.objective == "pred_noise":
      pred_x0 = self.predict_start_from_noise(x_t, tf, out)
    elif self.objective == "pred_x_start":
      pred_x0 = out
    else:
      raise ValueError(f"unknown objective {self.objective}")

    pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

    mean, std = self.q_posterior(pred_x0, x_t, tf)
    noise = torch.randn_like(x_t) if t > 0 else 0.0
    x_tm1: torch.Tensor = mean + std * noise
    ##################################################################

    return x_tm1

  @torch.no_grad()
  def sample(self, batch_size=16, return_all_timesteps=False, model_kwargs={}):

    shape = (batch_size, self.channels, self.image_size, self.image_size)
    img = torch.randn(shape, device=self.betas.device)  # type: ignore
    imgs = [img]

    for t in tqdm.asyncio.tqdm(
        reversed(range(0, self.num_timesteps)),
        desc="sampling loop time step",
        total=self.num_timesteps,
    ):
      img = self.p_sample(img, t, model_kwargs=model_kwargs)
      imgs.append(img)

    res = img if not return_all_timesteps else torch.stack(imgs, dim=1)
    res = self.unnormalize(res)
    return res

  def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
    """Sample from q(x_t | x_0) according to Eq. (4) of the paper.

    Args:
        x_start: (b, *) tensor. Starting image.
        t: (b,) tensor. Time step.
        noise: (b, *) tensor. Noise from N(0, 1).
    Returns:
        x_t: (b, *) tensor. Noisy image.
    """
    ####################################################################
    # TODO:
    # Implement sampling from q(x_t | x_0) according to Eq. (4) of the paper.
    # Hints: (1) Look at the `__init__` method to see precomputed coefficients.
    # (2) Use the `extract` function defined above to extract the coefficients
    # for the given time step `t`. (3) Recall that sampling from N(mu, sigma^2)
    # can be done as: x_t = mu + sigma * noise where noise is sampled from N(0, 1).
    # Approximately 3 lines of code.
    ####################################################################
    extracted: torch.Tensor = extract(
      self.sqrt_alphas_cumprod, t, x_start.shape)
    extracted_minus: torch.Tensor = extract(
      self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
    x_t: torch.Tensor = extracted * x_start + extracted_minus * noise
    return x_t

  def p_losses(self, x_start, model_kwargs={}):
    b, nts = x_start.shape[0], self.num_timesteps
    t = torch.randint(0, nts, (b,), device=x_start.device).long()  # (b,)
    x_start = self.normalize(x_start)  # (b, *)
    noise = torch.randn_like(x_start)  # (b, *)
    target = noise if self.objective == "pred_noise" else x_start  # (b, *)
    loss_weight = extract(self.loss_weight, t, target.shape)  # (b, *)
    ####################################################################
    # TODO:
    # Implement the loss function according to Eq. (14) of the paper.
    # First, sample x_t from q(x_t | x_0) using the `q_sample` function.
    # Then, get model predictions by calling self.model with appropriate args.
    # Finally, compute the weighted MSE loss.
    # Approximately 3-4 lines of code.
    ####################################################################
    x_t: torch.Tensor = self.q_sample(x_start, t, noise)
    out: torch.Tensor = self.model(x_t, t, model_kwargs)
    loss = torch.mean(loss_weight * (out - target) ** 2)
    ####################################################################

    return loss


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: tuple):
  """
  Extracts the appropriate coefficient values based on the given timesteps.

  This function gathers the values from the coefficient tensor `a` according to
  the given timesteps `t` and reshapes them to match the required shape such that
  it supports broadcasting with the tensor of given shape `x_shape`.

  Args:
      a (torch.Tensor): A tensor of shape (T,), containing coefficient values for all timesteps.
      t (torch.Tensor): A tensor of shape (b,), representing the timesteps for each sample in the batch.
      x_shape (tuple): The shape of the input image tensor, usually (b, c, h, w).

  Returns:
      torch.Tensor: A tensor of shape (b, 1, 1, 1), containing the extracted coefficient values
                    from a for corresponding timestep of each batch element, reshaped accordingly.
  """
  b, *_ = t.shape  # Extract batch size from the timestep tensor
  out = a.gather(-1, t)  # Gather the coefficient values from `a` based on `t`
  out = out.reshape(
      b, *((1,) * (len(x_shape) - 1))
  )  # Reshape to (b, 1, 1, 1) for broadcasting
  return out


def linear_beta_schedule(timesteps: int):
  """
  linear schedule, proposed in original ddpm paper
  """
  scale = 1000 / timesteps
  beta_start = scale * 0.0001
  beta_end = scale * 0.02
  return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps: int, s=0.008):
  """
  cosine schedule
  as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
  """
  steps = timesteps + 1
  t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
  alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
  alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
  betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
  return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps: int, start=-3, end=3, tau=1, clamp_min=1e-5):
  """
  sigmoid schedule
  proposed in https://arxiv.org/abs/2212.11972 - Figure 8
  better for images > 64x64, when used during training
  """
  steps = timesteps + 1
  t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
  v_start = torch.tensor(start / tau).sigmoid()
  v_end = torch.tensor(end / tau).sigmoid()
  alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (
      v_end - v_start
  )
  alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
  betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
  return torch.clip(betas, 0, 0.999)


def get_beta_schedule(beta_schedule: str, timesteps: int):
  if beta_schedule == "linear":
    beta_schedule_fn = linear_beta_schedule
  elif beta_schedule == "cosine":
    beta_schedule_fn = cosine_beta_schedule
  elif beta_schedule == "sigmoid":
    beta_schedule_fn = sigmoid_beta_schedule
  else:
    raise ValueError(f"unknown beta schedule {beta_schedule}")

  betas = beta_schedule_fn(timesteps)
  return betas
