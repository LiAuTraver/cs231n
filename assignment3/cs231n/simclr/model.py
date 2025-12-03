import torch
from torchvision.models.resnet import resnet50


class Model(torch.nn.Module):
  def __init__(self, feature_dim=128):
    super(Model, self).__init__()

    self.f = []  # type: ignore
    for name, module in resnet50().named_children():
      if name == 'conv1':
        # ResNet's original conv1 is designed for ImageNet (224×224 images), but CIFAR-10 uses 32×32 images.
        # too aggressive for small images, drastically reduce spatial dimensions too quickly.
        module = torch.nn.Conv2d(
          3, 64, kernel_size=3, stride=1, padding=1, bias=False)
      if not isinstance(module, torch.nn.Linear) and not isinstance(module, torch.nn.MaxPool2d):
        # maxpool downsampling aggressively for small datasets
        # linear module stored data of imagenet <- ResNet uses
        # ^^^ we are use them from ResNet, hence those are't suitable, drop them here
        # module's type:
        # torch.nn.modules.{conv.Conv2d, batchnorm.BatchNorm2d,
        # activation.ReLU, container.Sequential, pooling.AdaptiveAvgPool2d}
        # ^^^ feature extraction, activation, normalization, residuals, poolings can keep.
        self.f.append(module)
    # encoder
    self.f: torch.nn.Sequential = torch.nn.Sequential(*self.f)
    # projection head
    self.g = torch.nn.Sequential(torch.nn.Linear(2048, 512, bias=False),  # h has 2048 dim
                                 torch.nn.BatchNorm1d(512),
                                 torch.nn.ReLU(inplace=True),
                                 torch.nn.Linear(512, feature_dim, bias=True)
                                 )  # z has 128 dim

  def forward(self, x: torch.Tensor):
    x = self.f(x)
    feature = torch.flatten(x, start_dim=1)
    out = self.g(feature)
    return torch.nn.functional.normalize(feature, dim=-1), torch.nn.functional.normalize(out, dim=-1)
