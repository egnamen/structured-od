import copy
import math
import random

import numpy as np
import torch
from torchvision.transforms import transforms


class RescaleImageToSize(object):
    def __init__(self, size_to_use):
        self.size_to_use = size_to_use
        self.height_to_use, self.width_to_use = size_to_use
        self.resize_transform = transforms.Resize(self.size_to_use)

    def __call__(self, image, annotation=None):
        # Resize the image to wanted size
        resized_image = self.resize_transform(image)

        # If annotation is supplied we must also update it to match the operation performed to image
        if annotation and annotation["bbox"]:
            x_scale_change = self.width_to_use / image.width
            y_scale_change = self.height_to_use / image.height
            x, y, w, h = annotation["bbox"]

            annotation["bbox"] = [
                int(x * x_scale_change),
                int(y * y_scale_change),
                int(w * x_scale_change),
                int(h * y_scale_change),
            ]

        return resized_image, annotation


class RescaleShortestImageSideToSpecificLength(object):
    def __init__(self, size_to_use):
        self.size_to_use = size_to_use
        self.height_to_use, self.width_to_use = size_to_use

    def __call__(self, image, annotation):
        # Resize the image to wanted size
        if image.width < image.height:
            ratio = self.width_to_use / image.width
            resized_image = image.resize((self.width_to_use, math.ceil(image.height * ratio)))

        else:
            ratio = self.height_to_use / image.height
            resized_image = image.resize((math.ceil(image.width * ratio), self.height_to_use))

        # If annotation is supplied we must also update it to match the operation performed to image
        if annotation and annotation["bbox"]:
            x_scale_change = resized_image.width / image.width
            y_scale_change = resized_image.height / image.height
            x, y, w, h = annotation["bbox"]

            annotation["bbox"] = [
                int(x * x_scale_change),
                int(y * y_scale_change),
                int(w * x_scale_change),
                int(h * y_scale_change),
            ]

        return resized_image, annotation


class CropImageSquareByRandomChoiceTowardsBbox(object):
    def __init__(self, size_to_use):
        self.size_to_use = size_to_use
        self.height_to_use, self.width_to_use = size_to_use

    def __call__(self, image, annotation=None):
        # Extract the bounding box information
        x, y, w, h = annotation["bbox"]

        # Calculate the leftover pixels to crop
        leftover_pixels_to_crop = np.abs(image.width - image.height)
        pixels_to_crop_from_first_side = random.randint(0, leftover_pixels_to_crop)
        pixels_to_crop_from_second_side = leftover_pixels_to_crop - pixels_to_crop_from_first_side

        if image.width < image.height:
            # Crop height down towards width
            cropped_image = image.crop((0, pixels_to_crop_from_first_side, image.width, image.height - pixels_to_crop_from_second_side))

            # Update the annotation bbox
            pixels_removed_on_bbox_top = y - pixels_to_crop_from_first_side
            pixels_removed_on_bbox_bottom = (image.height - pixels_to_crop_from_second_side) - (y + h)

            new_y = 0 if pixels_removed_on_bbox_top < 0 else y - pixels_to_crop_from_first_side
            new_height = h
            new_height += pixels_removed_on_bbox_top if pixels_removed_on_bbox_top < 0 else 0
            new_height += pixels_removed_on_bbox_bottom if pixels_removed_on_bbox_bottom < 0 else 0

            annotation["bbox"] = [x, new_y, w, new_height]

        else:
            # Crop width down towards height
            cropped_image = image.crop((pixels_to_crop_from_first_side, 0, image.width - pixels_to_crop_from_second_side, image.height))

            # Update the annotation bbox
            pixels_removed_on_bbox_left = x - pixels_to_crop_from_first_side
            pixels_removed_on_bbox_right = (image.width - pixels_to_crop_from_second_side) - (x + w)

            new_x = 0 if pixels_removed_on_bbox_left < 0 else x - pixels_to_crop_from_first_side
            new_width = w
            new_width += pixels_removed_on_bbox_left if pixels_removed_on_bbox_left < 0 else 0
            new_width += pixels_removed_on_bbox_right if pixels_removed_on_bbox_right < 0 else 0

            annotation["bbox"] = [new_x, y, new_width, h]

        return cropped_image, annotation


class CropImageTowardsAnnotationBbox(object):
    def __init__(self, size_to_use):
        self.size_to_use = size_to_use
        self.height_to_use, self.width_to_use = size_to_use

    def __call__(self, image, annotation=None):
        # Extract the original bbox
        bbox = annotation["bbox"]

        # Calculate the leftover pixels to crop
        leftover_pixels_to_crop = np.abs(image.width - image.height)

        if image.width < image.height:
            # Crop towards the bounding box from the top
            image, pixels_moved_from_top, bbox = self._crop_top_side_towards_bbox(
                image, leftover_pixels_to_crop, bbox
            )

            # If there is still leftover pixels to crop we also crop from the bottom
            if leftover_pixels_to_crop > 0:
                image, pixels_moved_from_bottom, bbox = self._crop_bottom_side_towards_bbox(
                    image, leftover_pixels_to_crop, bbox
                )

        else:
            # Crop towards the bounding box from the left
            image, pixels_moved_from_left, bbox = self._crop_left_side_towards_bbox(
                image, leftover_pixels_to_crop, bbox
            )

            # If there is still leftover pixels to crop we also crop from the right
            if leftover_pixels_to_crop > 0:
                image, pixels_moved_from_right, bbox = self._crop_right_side_towards_bbox(
                    image, leftover_pixels_to_crop, bbox
                )

        # Update the annotation bbox
        annotation["bbox"] = bbox

        return image, annotation

    @staticmethod
    def _crop_left_side_towards_bbox(image, leftover_pixels_to_crop, bbox):
        x, y, w, h = bbox

        # Count how many pixels we have moved
        pixels_moved_from_left = 0

        # Set how much free space we have to crop
        croppable_pixels_from_left = x

        # Calculate the leftover pixels after cropping towards the bounding box from the left
        leftover_pixels_to_crop = croppable_pixels_from_left - leftover_pixels_to_crop

        # We have to crop as much as possible
        if leftover_pixels_to_crop <= 0:
            cropped_image = image.crop((x, 0, image.width, image.height))
            pixels_moved_from_left += x

        # We dont need to crop the entire way
        else:
            cropped_image = image.crop((leftover_pixels_to_crop, 0, image.width, image.height))
            pixels_moved_from_left += leftover_pixels_to_crop

        return cropped_image, pixels_moved_from_left, [x - pixels_moved_from_left, y, w, h]

    @staticmethod
    def _crop_right_side_towards_bbox(image, leftover_pixels_to_crop, bbox):
        x, y, w, h = bbox

        # Count how many pixels we have moved
        pixels_moved_from_right = 0

        # Set how much free space we have to crop
        croppable_pixels_from_right = image.width - (x + w)

        # Calculate the leftover pixels after cropping towards the bounding box from the left
        leftover_pixels_to_crop = croppable_pixels_from_right - leftover_pixels_to_crop

        # We have to crop as much as possible
        if leftover_pixels_to_crop <= 0:
            cropped_image = image.crop((0, 0, x + w, image.height))
            pixels_moved_from_right += image.width - (x + w)

        # We dont need to crop the entire way
        else:
            cropped_image = image.crop((0, 0, image.width - leftover_pixels_to_crop, image.height))
            pixels_moved_from_right += leftover_pixels_to_crop

        return cropped_image, pixels_moved_from_right, [x, y, w, h]

    @staticmethod
    def _crop_top_side_towards_bbox(image, leftover_pixels_to_crop, bbox):
        x, y, w, h = bbox

        # Count how many pixels we have moved
        pixels_moved_from_top = 0

        # Set how much free space we have to crop
        croppable_pixels_from_top = y

        # Calculate the leftover pixels after cropping towards the bounding box from the left
        leftover_pixels_to_crop = croppable_pixels_from_top - leftover_pixels_to_crop

        # We have to crop as much as possible
        if leftover_pixels_to_crop <= 0:
            cropped_image = image.crop((0, y, image.width, image.height))
            pixels_moved_from_top += y

        # We dont need to crop the entire way
        else:
            cropped_image = image.crop((0, leftover_pixels_to_crop, image.width, image.height))
            pixels_moved_from_top += leftover_pixels_to_crop

        return cropped_image, pixels_moved_from_top, [x, y - pixels_moved_from_top, w, h]

    @staticmethod
    def _crop_bottom_side_towards_bbox(image, leftover_pixels_to_crop, bbox):
        x, y, w, h = bbox

        # Count how many pixels we have moved
        pixels_moved_from_bottom = 0

        # Set how much free space we have to crop
        croppable_pixels_from_bottom = image.height - (y + h)

        # Calculate the leftover pixels after cropping towards the bounding box from the left
        leftover_pixels_to_crop = croppable_pixels_from_bottom - leftover_pixels_to_crop

        # We have to crop as much as possible
        if leftover_pixels_to_crop <= 0:
            cropped_image = image.crop((0, 0, image.width, y + h))
            pixels_moved_from_bottom += image.height - (y + h)

        # We dont need to crop the entire way
        else:
            cropped_image = image.crop((0, 0, image.width, image.height - leftover_pixels_to_crop))
            pixels_moved_from_bottom += leftover_pixels_to_crop

        return cropped_image, pixels_moved_from_bottom, [x, y, w, h]


class VGG16HardCroppingRandomlyCompose(object):
    def __init__(self, device=torch.device("cpu")):
        self.size_to_use = (224, 224)
        self.height_to_use, self.width_to_use = self.size_to_use
        self.device = device

        self.shortest_side_rescale_transform = RescaleShortestImageSideToSpecificLength(
            self.size_to_use
        )
        self.cropping_towards_annotation_transform = CropImageTowardsAnnotationBbox(
            self.size_to_use
        )
        self.cropping_image_square_random = CropImageSquareByRandomChoiceTowardsBbox(
            self.size_to_use
        )

        self.rescale_transform = RescaleImageToSize(self.size_to_use)
        self.tensor_transform = transforms.ToTensor()
        self.device_transform = transforms.Lambda(lambda tensor: tensor.to(self.device))

    def __call__(self, image, annotation=None):
        # Transform the image to the proper scale
        image, annotation = self.shortest_side_rescale_transform(image, annotation)
        image, annotation = self.cropping_towards_annotation_transform(image, annotation)
        image, annotation = self.cropping_image_square_random(image, annotation)

        # Transform image to tensor data and move to the proper device
        tensor_data = self.tensor_transform(image)
        tensor_data = self.device_transform(tensor_data)

        # Transform annotation to tensor data and move to the proper device.
        # Because some annotations might be missing or uncastable we only try once and set anything that fails to None
        if annotation:
            annotation = copy.deepcopy(annotation)
            for key, value in annotation.items():
                try:
                    annotation[key] = torch.tensor(value, device=self.device)
                except Exception:
                    annotation[key] = None

        return tensor_data, annotation


class VGG16SoftCroppingThenHardRescaleCompose(object):
    def __init__(self, device=torch.device("cpu")):
        self.size_to_use = (224, 224)
        self.height_to_use, self.width_to_use = self.size_to_use
        self.device = device

        self.shortest_side_rescale_transform = RescaleShortestImageSideToSpecificLength(self.size_to_use)
        self.cropping_towards_annotation_transform = CropImageTowardsAnnotationBbox(self.size_to_use)

        self.rescale_transform = RescaleImageToSize(self.size_to_use)
        self.tensor_transform = transforms.ToTensor()
        self.device_transform = transforms.Lambda(lambda tensor: tensor.to(self.device))

    def __call__(self, image, annotation=None):
        # Transform the image to the proper scale
        image, annotation = self.shortest_side_rescale_transform(image, annotation)
        image, annotation = self.cropping_towards_annotation_transform(image, annotation)
        image, annotation = self.rescale_transform(image, annotation)

        # Transform image to tensor data and move to the proper device
        tensor_data = self.tensor_transform(image)
        tensor_data = self.device_transform(tensor_data)

        # Transform annotation to tensor data and move to the proper device.
        # Because some annotations might be missing or uncastable we only try once and set anything that fails to None
        if annotation:
            annotation = copy.deepcopy(annotation)
            for key, value in annotation.items():
                try:
                    annotation[key] = torch.tensor(value, device=self.device)
                except Exception:
                    annotation[key] = None

        return tensor_data, annotation


class VGG16HardRescaleCompose(object):
    def __init__(self, device=torch.device("cpu")):
        self.size_to_use = (224, 224)
        self.height_to_use, self.width_to_use = self.size_to_use
        self.device = device

        self.shortest_side_rescale_transform = RescaleShortestImageSideToSpecificLength(self.size_to_use)

        self.rescale_transform = RescaleImageToSize(self.size_to_use)
        self.tensor_transform = transforms.ToTensor()
        self.device_transform = transforms.Lambda(lambda tensor: tensor.to(self.device))

    def __call__(self, image, annotation=None):
        # Transform the image to the proper scale
        image, annotation = self.shortest_side_rescale_transform(image, annotation)
        image, annotation = self.rescale_transform(image, annotation)

        # Transform image to tensor data and move to the proper device
        tensor_data = self.tensor_transform(image)
        tensor_data = self.device_transform(tensor_data)

        # Transform annotation to tensor data and move to the proper device.
        # Because some annotations might be missing or uncastable we only try once and set anything that fails to None
        if annotation:
            annotation = copy.deepcopy(annotation)
            for key, value in annotation.items():
                try:
                    annotation[key] = torch.tensor(value, device=self.device)
                except Exception:
                    annotation[key] = None

        return tensor_data, annotation
