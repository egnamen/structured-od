import torch

from torch.nn import Parameter, ModuleList

from abc import ABC, abstractmethod
from torchvision.models import vgg


class LogisticFunction(torch.nn.Module):
    def __init__(self, intercept, rate):
        super().__init__()
        self.intercept = Parameter(torch.as_tensor(intercept), requires_grad=False)
        self.rate = Parameter(torch.as_tensor(rate), requires_grad=False)

    def __repr__(self):
        return f"Logistic(intercept=({float(self.intercept):.4f}), rate=({float(self.rate):.4f}))"

    def forward(self, x):
        return 1 / (1 + torch.exp(-(self.intercept + self.rate * x)))


class PartDetectorHead(torch.nn.Module):
    def __init__(self, filter_number, convolution_module, logistic_function):
        super().__init__()
        self.filter_number = filter_number
        self.convolution_module = convolution_module
        self.activation_function = logistic_function

    def forward(self, x):
        x = self.convolution_module(x)
        x = self.activation_function(x)

        return x[:, 0, :, :]


class AbstractPartsExtractor(ABC, torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = self.initialize_backbone()
        self.backbone = self.dissect_backbone(self.backbone)

    @staticmethod
    @abstractmethod
    def initialize_backbone():
        pass

    @abstractmethod
    def dissect_backbone(self, backbone):
        pass

    def forward(self, activations):
        for module in self.backbone:
            activations = module(activations)

        return activations


class PartsExtractorVGG16(AbstractPartsExtractor):
    def __init__(self, from_layer_number):
        self.from_layer_number = from_layer_number
        super().__init__()

    @staticmethod
    def initialize_backbone():
        return vgg.vgg16(weights=vgg.VGG16_Weights.IMAGENET1K_V1).features

    def dissect_backbone(self, backbone):
        dissection_layers = torch.nn.ModuleList()
        backbone_layers = list(backbone.children())

        end_slice = self.from_layer_number + 1
        for index, layer in enumerate(backbone_layers[:end_slice]):
            dissection_layers.add_module(name=f"layer_{index}", module=layer)

        return torch.nn.Sequential(*dissection_layers)


class PartDetectorsModel(torch.nn.Module):
    def __init__(self, extractor_backbone, part_detection_heads):
        super().__init__()
        self.extractor_backbone = extractor_backbone
        self.part_detection_heads = ModuleList(part_detection_heads)

    def forward(self, x):
        x = self.extractor_backbone(x)

        y = []
        for part_detector_head in self.part_detection_heads:
            y.append(part_detector_head(x))
        y = torch.stack(y).moveaxis(0, 1)

        return y
