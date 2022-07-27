import torch
import yaml
import copy
import numpy as np

from tqdm import tqdm
from scipy import stats
from sklearn.linear_model import LogisticRegression

from sources.dataset import BinaryLabeledCOCODataset
from sources.dataset.transform import VGG16HardRescaleCompose
from sources.model import PartsExtractorVGG16, LogisticFunction, PartDetectorHead, PartDetectorsModel
from sources.constant import GLOBAL_SEED, POSITIVE_CATEGORY, NEGATIVE_CATEGORY


def parse_config_file(config_file_path):
    with open(config_file_path, "r") as config_file:
        config = yaml.safe_load(config_file)

    return config


def create_dataset(config):
    device_to_use = torch.device(config["training"]["device"])

    if config["dataset"]["class_name"] == "BinaryLabeledCOCODataset":
        return BinaryLabeledCOCODataset(
            image_directory_path=config["dataset"]["attributes"]["image_directory_path"],
            annotation_path=config["dataset"]["attributes"]["annotation_file_path"],
            transformation=VGG16HardRescaleCompose(device=device_to_use)
        )

    raise ValueError("Unexpected value supplied in 'dataset:class_name' key of config")


def create_dataloader(config, dataset):
    return torch.utils.data.DataLoader(
        dataset=dataset,
        shuffle=config["loader"]["shuffle"],
        batch_size=config["loader"]["batch_size"],
        collate_fn=dataset.collate_fn,
        num_workers=0,
    )


def __create_parts_extractor(config):
    print("> Creating parts extractor...")

    device_to_use = torch.device(config["training"]["device"])

    if config["model"]["class_name"] == "VGG16PartDetectors":
        return PartsExtractorVGG16(
            from_layer_number=config["model"]["attributes"]["from_layer_number"]
        ).to(device=device_to_use)

    raise ValueError("Unexpected value supplied in 'model:class_name' key of config")


def create_part_detectors_model(config, loader):
    parts_extractor = __create_parts_extractor(config)

    feature_maximums, categories = __measure_feature_maximums(config, parts_extractor, loader)

    sliced_parts_extractor, output_convolution_module = parts_extractor.backbone[:-1], parts_extractor.backbone[-1]
    part_detector_heads = create_part_detector_heads(config, output_convolution_module, feature_maximums, categories)

    part_detectors_model = PartDetectorsModel(
        extractor_backbone=sliced_parts_extractor,
        part_detection_heads=part_detector_heads,
    )

    device_to_use = torch.device(config["training"]["device"])
    part_detectors_model = part_detectors_model.to(device_to_use)

    return part_detectors_model


def create_part_detector_heads(config, output_convolution_module, feature_maximums, categories):
    print("> Creating part detector heads...")

    signed_distances = __calculate_distances(feature_maximums, categories)
    part_detector_filters = __calculate_part_detector_filters(signed_distances, config["model"]["attributes"]["cdf_threshold"])

    part_detector_heads = []
    for part_detector_filter in part_detector_filters:
        filter_convolution_module = __create_filter_convolution_module(output_convolution_module, part_detector_filter)

        filter_feature_maximums = feature_maximums[:, part_detector_filter]
        filter_logistic_function = __calculate_filter_logistic_function(filter_feature_maximums, categories)

        part_detector_head = PartDetectorHead(
            filter_number=part_detector_filter,
            convolution_module=filter_convolution_module,
            logistic_function=filter_logistic_function
        )

        part_detector_heads.append(part_detector_head)

    return part_detector_heads


def __calculate_part_detector_filters(signed_distances, cdf_threshold):
    mu, sigma = stats.norm.fit(signed_distances)
    cdf_values = stats.norm.cdf(signed_distances, mu, sigma)

    filter_indexation = torch.as_tensor(cdf_values > cdf_threshold, dtype=torch.bool)
    part_detector_filters, = torch.where(filter_indexation)

    return part_detector_filters


def __get_maximum_iterations(config, loader):
    if "maximum_iterations" in config["training"].keys() and config["training"]["maximum_iterations"] is not None:
        maximum_iterations = config["training"]["maximum_iterations"]
    else:
        maximum_iterations = len(loader)

    return maximum_iterations


def __measure_feature_maximums(config, parts_extractor, loader):
    print("> Measuring feature maximums...")

    feature_maxiums, categories = [], []

    maximum_iterations = __get_maximum_iterations(config, loader)
    progress_bar = tqdm(total=maximum_iterations, unit="batch")
    for iteration, (data_batch, annotation_batch) in enumerate(loader):
        if iteration > maximum_iterations:
            break

        feature_maps_batch = dcn(parts_extractor(data_batch))
        feature_maxiums_batch = feature_maps_batch.max(axis=(2, 3))
        feature_maxiums.append(feature_maxiums_batch)

        categories_batch = [dcn(annotation["category_id"]) for annotation in annotation_batch]
        categories.append(categories_batch)

        progress_bar.set_postfix(dict({"Iteration": f"{iteration}"}), refresh=True)
        progress_bar.update()
    progress_bar.close()

    # Reduce list of batches into single numpy array over all forward passes
    feature_maxiums = np.concatenate(feature_maxiums, axis=0)
    categories = np.concatenate(categories, axis=0)

    return feature_maxiums, categories


def __calculate_distances(feature_maximums, categories):
    print("> Calculating distance of filters...")

    signed_distances = []
    for filter_feature_maximums in np.moveaxis(feature_maximums, 1, 0):

        positive_feature_maximums = filter_feature_maximums[categories == POSITIVE_CATEGORY]
        positive_mu, positive_sigma = stats.norm.fit(positive_feature_maximums)

        negative_feature_maximums = filter_feature_maximums[categories == NEGATIVE_CATEGORY]
        negative_mu, negative_sigma = stats.norm.fit(negative_feature_maximums)

        direction = 1 if positive_mu > negative_mu else -1
        distance = __calculate_bhattacharyya_distance(positive_mu, positive_sigma, negative_mu, negative_sigma)
        signed_distance = direction * distance

        signed_distances.append(signed_distance)
    signed_distances = np.array(signed_distances)

    return signed_distances


def __calculate_bhattacharyya_distance(mu_1, sigma_1, mu_2, sigma_2):
    return 0.25 * (
        np.log(0.25 * ((sigma_1**2 / sigma_2**2) + (sigma_2**2 / sigma_1**2) + 2))
        + ((mu_1 - mu_2) ** 2 / (sigma_1**2 + sigma_2**2))
    )


def __create_filter_convolution_module(template_convolution_module, filter_number):
    filter_convolution_module = copy.deepcopy(template_convolution_module)

    filter_weight = filter_convolution_module.weight[filter_number, :, :, :].unsqueeze(dim=0)
    filter_convolution_module.weight = torch.nn.Parameter(filter_weight)

    filter_bias = filter_convolution_module.bias[filter_number].unsqueeze(dim=0)
    filter_convolution_module.bias = torch.nn.Parameter(filter_bias)

    return filter_convolution_module


def __calculate_filter_logistic_function(filter_feature_maximums, categories):
    filter_feature_maximums = np.expand_dims(filter_feature_maximums, 1)
    regressor = LogisticRegression(random_state=GLOBAL_SEED).fit(filter_feature_maximums, categories)
    logistic_function = LogisticFunction(intercept=regressor.intercept_, rate=regressor.coef_)

    return logistic_function


def convert_tensor_to_image(tensor):
    if tensor.ndim == 4:
        image_batch = np.moveaxis(dcn(tensor), 1, -1)
        return image_batch

    elif tensor.ndim == 3:
        image = np.moveaxis(dcn(tensor), 0, -1)
        return image

    raise ValueError("The 'tensor' parameter did not have the expected amount of dimensions")


def dcn(tensor):
    return tensor.detach().cpu().numpy()
