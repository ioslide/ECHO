"""Backbones and preprocessing used by the released ImageNet experiments."""
import torch
from torch import nn
import torchvision
from packaging import version
from loguru import logger as log
from robustbench.model_zoo.architectures.utils_architectures import normalize_model, ImageNormalizer
from . import resnet as Resnet
from .imagenet_subsets import IMAGENET_D109_MASK

class ImageNetXMaskingLayer(nn.Module):
    """ Following: https://github.com/hendrycks/imagenet-r/blob/master/eval.py
    """
    def __init__(self, mask):
        super().__init__()
        self.mask = mask

    def forward(self, x):
        return x[:, self.mask]

class ImageNetXWrapper(torch.nn.Module):
    def __init__(self, model, mask):
        super().__init__()
        self.__dict__ = model.__dict__.copy()

        self.masking_layer = ImageNetXMaskingLayer(mask)

    def forward(self, x):
        logits = self.model(self.normalize(x))
        return self.masking_layer(logits)

def get_torchvision_model(model_name: str, weight_version: str = "IMAGENET1K_V1"):
    assert version.parse(torchvision.__version__) >= version.parse("0.13"), "Torchvision version has to be >= 0.13"

    # check if the specified model name is available in torchvision
    available_models = torchvision.models.list_models(module=torchvision.models)
    if model_name not in available_models:
        raise ValueError(f"Model '{model_name}' is not available in torchvision. Choose from: {available_models}")

    # get the weight object of the specified model and the available weight initialization names
    model_weights = torchvision.models.get_model_weights(model_name)
    available_weights = [init_name for init_name in dir(model_weights) if "IMAGENET1K" in init_name]

    # check if the specified type of weights is available
    if weight_version not in available_weights:
        raise ValueError(f"Weight type '{weight_version}' is not supported for torchvision model '{model_name}'."
                         f" Choose from: {available_weights}")

    # restore the specified weights
    model_weights = getattr(model_weights, weight_version)

    # setup the specified model and initialize it with the specified pre-trained weights
    model = torchvision.models.get_model(model_name, weights=model_weights)

    # get the transformation and add the input normalization to the model
    transform = model_weights.transforms()
    model = normalize_model(model, transform.mean, transform.std)
    log.info(f"Successfully restored '{weight_version}' pre-trained weights"
                f" for model '{model_name}' from torchvision!")

    return model


def build_model(cfg):
    if cfg.CORRUPTION.DATASET == "imagenet_d109":
        model = ImageNetXWrapper(get_torchvision_model(cfg.MODEL.ARCH), IMAGENET_D109_MASK)
    elif cfg.MODEL.ARCH in {"resnet18", "resnet34", "resnet50", "resnet101", "resnet152"}:
        model = getattr(Resnet, cfg.MODEL.ARCH)(pretrained=True, num_classes=cfg.CORRUPTION.NUM_CLASS)
    else:
        model = get_torchvision_model(cfg.MODEL.ARCH)
    return model.cuda()
