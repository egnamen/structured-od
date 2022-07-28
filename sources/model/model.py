import torch
import numpy as np

from abc import ABC, abstractmethod
from torch.nn import Parameter, ModuleList
from torchvision.models import vgg
from sklearn import tree

from sources.constant import GLOBAL_SEED


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


class AbstractPartsExtractorModel(ABC, torch.nn.Module):
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

    def forward(self, x):
        for module in self.backbone:
            x = module(x)

        return x


class VGG16PartsExtractorModel(AbstractPartsExtractorModel):
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

        all_x = []
        for part_detector_head in self.part_detection_heads:
            all_x.append(part_detector_head(x))
        all_x = torch.stack(all_x).moveaxis(0, 1)

        return all_x


class AbstractJointStructureModel(ABC, torch.nn.Module):
    def __init__(self, part_detectors_model, combination_model):
        super().__init__()
        self.part_detectors_model = part_detectors_model
        self.combination_model = combination_model

    @abstractmethod
    def preprocess_part_detections(self, x):
        pass

    @abstractmethod
    def predict_combination_model(self, x):
        pass

    @abstractmethod
    def fit_combination_model(self, x, y):
        pass

    def forward(self, x):
        x = self.part_detectors_model(x)
        x = self.preprocess_part_detections(x)
        x = self.predict_combination_model(x)

        return x


class DTJointStructureClassifier(AbstractJointStructureModel):
    def __init__(self, part_detectors_model, probability_threshold, max_depth, min_samples_split):
        self.probability_threshold = probability_threshold
        super().__init__(
            part_detectors_model=part_detectors_model,
            combination_model=tree.DecisionTreeClassifier(
                random_state=GLOBAL_SEED,
                max_depth=max_depth,
                min_samples_split=min_samples_split
            )
        )

    def preprocess_part_detections(self, x):
        probability_maps_batch = x
        batch_size, filter_size, _, _ = probability_maps_batch.shape

        segmentation_maps_batch = probability_maps_batch.amax(dim=1)
        segmentation_maps_batch = segmentation_maps_batch.detach().cpu().numpy().astype(np.float64)

        part_hits_maps_batch = probability_maps_batch.argmax(dim=1)
        part_hits_maps_batch = part_hits_maps_batch.detach().cpu().numpy().astype(np.float64)
        part_hits_maps_batch[segmentation_maps_batch < self.probability_threshold] = np.nan

        flattened_part_hits_batch = part_hits_maps_batch.reshape(batch_size, -1)
        histogram_function = lambda array: np.histogram(array, bins=np.arange(0, filter_size + 1))[0]
        histograms_batch = np.apply_along_axis(histogram_function, 1, flattened_part_hits_batch)

        return histograms_batch

    def predict_combination_model(self, x):
        return self.combination_model.predict(x)

    def fit_combination_model(self, x, y):
        self.combination_model = self.combination_model.fit(x, y)
