import torch

from torch.utils.data import Dataset
from typing import Tuple
from pycocotools import coco
from PIL import Image


class BinaryLabeledCOCODataset(Dataset):
    """
    Dataset consisting of images annotated with COCO format.
    All annotations labels are expected to be binary.
    """

    def __init__(self, image_directory_path, annotation_path, transformation=None):
        self.image_directory_path = image_directory_path
        self.annotation_path = annotation_path
        self.transformation = transformation
        self.coco_object = coco.COCO(annotation_file=self.annotation_path)
        self.image_ids = sorted(self.coco_object.imgs.keys())

    @staticmethod
    def collate_fn(dataset_batch):
        # Batching is set to a 'int' -> Return 'data_batch' and 'annotation_batch' tuple
        if type(dataset_batch) == list:
            return (
                torch.stack([data for data, _ in dataset_batch]),
                [annotation for _, annotation in dataset_batch]
            )

        # Batching is set to 'None' -> Return 'data, annotation' tuple
        elif type(dataset_batch) == tuple:
            return dataset_batch
        else:
            raise ValueError("Unexpected type appeared for dataset batch")

    def load_image(self, image_id) -> Tuple[Image.Image, str]:
        image_dictionary = self.coco_object.loadImgs([image_id])[0]
        image_path = self.image_directory_path + "/" + image_dictionary["file_name"]

        return Image.open(image_path).convert("RGB")

    def load_annotation(self, image_id) -> dict:
        annotation_ids = self.coco_object.getAnnIds(image_id)
        annotations = self.coco_object.loadAnns(annotation_ids)

        if len(annotations) > 1:
            raise ValueError(
                "Dataset assumes each image corresponds to a single annotation, "
                "but multiple were loaded for image_id={number}".format(number=image_id)
            )

        return annotations[0]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]

        image = self.load_image(image_id)
        annotation = self.load_annotation(image_id)

        if self.transformation:
            image, annotation = self.transformation(image, annotation)

        return image, annotation
