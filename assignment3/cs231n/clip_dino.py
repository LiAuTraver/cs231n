from typing import Callable
from tensorflow.python.framework.ops import device_v2
import tensorflow_datasets
import torch
import numpy as np
import clip
from PIL import Image
import cv2
import torchvision


def get_similarity_no_loop(text_features: torch.Tensor, image_features: torch.Tensor):
  """
  Computes the pairwise cosine similarity between text and image feature vectors.

  Args:
      text_features (torch.Tensor): A tensor of shape (N, D).
      image_features (torch.Tensor): A tensor of shape (M, D).

  Returns:
      torch.Tensor: A similarity matrix of shape (N, M), where each entry (i, j)
      is the cosine similarity between text_features[i] and image_features[j].
  """
  ############################################################################
  # TODO: Compute the cosine similarity. Do NOT use for loops.               #
  ############################################################################
  text_norm = text_features / text_features.norm(dim=1, keepdim=True)
  image_norm = image_features / image_features.norm(dim=1, keepdim=True)
  similarity = text_norm @ image_norm.T
  ############################################################################
  #                             END OF YOUR CODE                             #
  ############################################################################

  return similarity


@torch.no_grad()
def clip_zero_shot_classifier(clip_model: torch.nn.Module, clip_preprocess: Callable, images: list[np.ndarray],
                              class_texts: list[str], device: str | torch.device):
  """Performs zero-shot image classification using a CLIP model.

  Args:
      clip_model (torch.nn.Module): The pre-trained CLIP model for encoding
          images and textorchvision.transforms.
      clip_preprocess (Callable): A preprocessing function to apply to each
          image before encoding.
      images (List[np.ndarray]): A list of input images as NumPy arrays
          (H x W x C) uint8.
      class_texts (List[str]): A list of class label strings for zero-shot
          classification.
      device (torch.device): The device on which computation should be
          performed. Pass text_tokens to this device before passing it to
          clip_model.

  Returns:
      List[str]: Predicted class label for each image, selected from the
          given class_texts.
  """

  pred_classes: list[str] = []

  ############################################################################
  # TODO: Find the class labels for images.                                  #
  ############################################################################
  # preprocess, encode images
  preprocessed_images = torch.stack(
      [clip_preprocess(Image.fromarray(image)) for image in images]).to(device)
  image_features = clip_model.encode_image(preprocessed_images)  # type: ignore
  # tokenize, encode class texts
  text_tokens = clip.tokenize(class_texts).to(device)
  text_features = clip_model.encode_text(text_tokens)  # type: ignore
  similarity = get_similarity_no_loop(text_features, image_features)
  # predicted classes
  predicted_indices = similarity.argmax(dim=0)
  pred_classes = [class_texts[index]
                  for index in predicted_indices.cpu().numpy()]
  ############################################################################
  #                             END OF YOUR CODE                             #
  ############################################################################

  return pred_classes


class CLIPImageRetriever:
  """
  A simple image retrieval system using CLIP.
  """

  @torch.no_grad()
  def __init__(self, clip_model: torch.nn.Module, clip_preprocess: Callable, images: list[np.ndarray], device: torch.device | str):
    """
    Args:
      clip_model (torch.nn.Module): The pre-trained CLIP model.
      clip_preprocess (Callable): Function to preprocess images.
      images (List[np.ndarray]): List of images as NumPy arrays (H x W x C).
      device (torch.device): The device for model execution.
    """
    ############################################################################
    # TODO: Store all necessary object variables to use in retrieve method.    #
    # Note that you should process all images at once here and avoid repeated  #
    # computation for each text query. You may end up NOT using the above      #
    # similarity function for most compute-optimal implementation.#
    ############################################################################
    self.clip_model = clip_model
    self.clip_preprocess = clip_preprocess
    self.device = device
    # preprocess and encode all images
    preprocessed_images = torch.stack(
        [clip_preprocess(Image.fromarray(image)) for image in images]).to(device)
    self.image_features = clip_model.encode_image(
      preprocessed_images)  # type: ignore

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

  @torch.no_grad()
  def retrieve(self, query: str, k: int = 2):
    """
    Retrieves the indices of the top-k images most similar to the input textorchvision.transforms.
    You may find torch.Tensor.topk method useful.

    Args:
        query (str): The text query.
        k (int): Return top k images.

    Returns:
        List[int]: Indices of the top-k most similar images.
    """
    top_indices = []
    ############################################################################
    # TODO: Retrieve the indices of top-k images.                              #
    ############################################################################
    # tokenize and encode query
    text_tokens = clip.tokenize([query]).to(self.device)
    text_features = self.clip_model.encode_text(text_tokens)  # type: ignore
    # compute similarity
    similarity: torch.Tensor = get_similarity_no_loop(
      text_features, self.image_features)
    # get top-k indices
    topk = similarity.topk(k=k, dim=1)
    top_indices: list[int] = topk.indices[0].cpu().numpy().tolist()
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    return top_indices


class DavisDataset:
  def __init__(self):
    self.davis = tensorflow_datasets.load(
      'davis/480p', split='validation', as_supervised=False)
    self.img_tsfm = torchvision.transforms.Compose([
        torchvision.transforms.Resize(
          (480, 480)), torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
          (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

  def get_sample(self, index):
    assert index < len(self.davis)
    ds_iter = iter(tensorflow_datasets.as_numpy(self.davis))
    for _ in range(index + 1):
      video = next(ds_iter)
    frames = video['video']['frames']  # type: ignore
    masks = video['video']['segmentations']  # type: ignore
    video_name = video['metadata']['video_name']  # type: ignore
    print(
      f"video {video_name.decode()}  {len(frames)} frames")
    return frames, masks

  def process_frames(self, frames, dino_model, device):
    res = []
    for f in frames:
      f = self.img_tsfm(Image.fromarray(f))[None].to(device)  # type: ignore
      with torch.no_grad():
        tok = dino_model.get_intermediate_layers(f, n=1)[0]
      res.append(tok[0, 1:])

    res = torch.stack(res)
    return res

  def process_masks(self, masks, device):
    res = []
    for m in masks:
      m = cv2.resize(m, (60, 60), cv2.INTER_NEAREST)  # type: ignore
      res.append(torch.from_numpy(m).long().flatten(-2, -1))
    res = torch.stack(res).to(device)
    return res

  def mask_frame_overlay(self, processed_mask: torch.Tensor, frame: np.ndarray) -> np.ndarray:
    H, W = frame.shape[:2]
    mask = processed_mask.detach().cpu().numpy()
    mask = mask.reshape((60, 60))
    mask = cv2.resize(
        mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    overlay = create_segmentation_overlay(mask, frame.copy())
    return overlay


def create_segmentation_overlay(segmentation_mask: np.ndarray, image: np.ndarray, alpha: float = 0.5) -> np.ndarray:
  """
  Generate a colored segmentation overlay on top of an RGB image.

  Parameters:
      segmentation_mask (np.ndarray): 2D array of shape (H, W), with class indices.
      image (np.ndarray): 3D array of shape (H, W, 3), RGB image.
      alpha (float): Transparency factor for overlay (0 = only image, 1 = only mask).

  Returns:
      np.ndarray: Image with segmentation overlay (shape: (H, W, 3), dtype: uint8).
  """
  assert segmentation_mask.shape[:2] == image.shape[:
                                                    2], "Segmentation and image size mismatch"
  assert image.dtype == np.uint8, "Image must be of type uint8"

  # Generate deterministic colors for each class using a fixed colormap
  def generate_colormap(n):
    np.random.seed(42)  # For determinism
    colormap = np.random.randint(0, 256, size=(n, 3), dtype=np.uint8)
    return colormap

  colormap = generate_colormap(10)

  # Create a color image for the segmentation mask
  seg_color = colormap[segmentation_mask]  # shape: (H, W, 3)

  # Blend with original image
  overlay = cv2.addWeighted(image, 1 - alpha, seg_color, alpha, 0)

  return overlay


def compute_iou(pred: torch.Tensor, gt: torch.Tensor, num_classes: int):
  """Compute the mean Intersection over Union (IoU)."""
  iou = 0
  for ci in range(num_classes):
    p = pred == ci
    g = gt == ci
    iou += (p & g).sum() / ((p | g).sum() + 1e-8)
  return iou / num_classes


class DINOSegmentation:
  def __init__(self, device: str | torch.device, num_classes: int, inp_dim: int = 384):
    """
    Initialize the DINOSegmentation model.

    This defines a simple neural network designed to  classify DINO feature
    vectors into segmentation classes. It includes model initialization,
    optimizer, and loss function setup.

    Args:
        device (torch.device): Device to run the model on (CPU or CUDA).
        num_classes (int): Number of segmentation classes.
        inp_dim (int, optional): Dimensionality of the input DINO features.
    """

    ############################################################################
    # TODO: Define a very lightweight pytorch model, optimizer, and loss       #
    # function to train classify each DINO feature vector into a seg. class.   #
    # It can be a linear layer or two layer neural network.                    #
    ############################################################################
    self.device = device
    self.num_classes = num_classes
    self.model = torch.nn.Linear(inp_dim, num_classes).to(device)
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
    self.criterion = torch.nn.CrossEntropyLoss()
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

  def train(self, X_train: torch.Tensor, Y_train: torch.Tensor, num_iters=500):
    """Train the segmentation model using the provided training data.

    Args:
        X_train (torch.Tensor): Input feature vectors of shape (N, D).
        Y_train (torch.Tensor): Ground truth labels of shape (N,).
        num_iters (int, optional): Number of optimization steps.
    """
    ############################################################################
    # TODO: Train your model for `num_iters` steps.                            #
    ############################################################################
    self.model.train()
    for _ in range(num_iters):
      self.optimizer.zero_grad()
      outputs = self.model(X_train)
      loss = self.criterion(outputs, Y_train)
      loss.backward()
      self.optimizer.step()
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

  @torch.no_grad()
  def inference(self, X_test: torch.Tensor):
    """Perform inference on the given test DINO feature vectors.

    Args:
        X_test (torch.Tensor): Input feature vectors of shape (N, D).

    Returns:
        torch.Tensor of shape (N,): Predicted class indices.
    """
    pred_classes = None
    ############################################################################
    # TODO: Train your model for `num_iters` steps.                            #
    ############################################################################
    self.model.eval()
    outputs = self.model(X_test)
    pred_classes = outputs.argmax(dim=1)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    return pred_classes
