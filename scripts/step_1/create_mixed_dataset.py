import numpy as np
import argparse
import os
import random
import math
import pathlib
import shutil

from distutils.dir_util import copy_tree
from datetime import date

from sources.data import utilities as data_utilities
from sources.dataset import utilities as dataset_utilities
from sources.constant import GLOBAL_SEED, POSITIVE_CATEGORY, NEGATIVE_CATEGORY

random_engine = random.Random(GLOBAL_SEED)


def main(arguments):
    if dataset_utilities.is_existing_dataset_directory(arguments.output_folder_name):
        raise RuntimeError("Output folder name provided in script arguments already exists")

    if (not os.path.exists(arguments.positive_data_directory_path)) or (not os.path.exists(arguments.negative_data_directory_path)):
        raise RuntimeError("Data path provided in the script arguments did not exists")

    temporary_directory_path = data_utilities.setup_temporary_data_directory()

    positive_image_annotations, negative_image_annotations = merge_images_to_directory(
        target_directory_path=temporary_directory_path,
        positive_data_directory_path=arguments.positive_data_directory_path,
        negative_data_directory_path=arguments.negative_data_directory_path
    )

    merged_coco_annotation = create_merged_annotation(
        positive_image_annotations=positive_image_annotations,
        negative_image_annotations=negative_image_annotations
    )

    __enforce_class_balance(
        image_directory_path=temporary_directory_path,
        coco_annotation=merged_coco_annotation
    )

    dataset_utilities.create_partitioned_dataset(
        dataset_name=arguments.output_folder_name,
        coco_annotation=merged_coco_annotation,
        image_directory_path=temporary_directory_path,
        testing_fraction=arguments.testing_fraction,
        validation_fraction=arguments.validation_fraction
    )

    shutil.rmtree(temporary_directory_path)


def merge_images_to_directory(target_directory_path, positive_data_directory_path, negative_data_directory_path):
    print("> Copying both data directories to common folder and renaming images uniquely...")

    positive_image_annotations = data_utilities.load_annotation_from_data_path(positive_data_directory_path)["images"]
    negative_image_annotations = data_utilities.load_annotation_from_data_path(negative_data_directory_path)["images"]

    number_of_positive_images, number_of_negative_images = len(positive_image_annotations), len(negative_image_annotations)
    number_of_leading_zeros = max(__length_of_number(number_of_positive_images), __length_of_number(number_of_negative_images))

    print("> Copying positive data directory...")
    copied_positive_file_paths = copy_tree(os.path.join(positive_data_directory_path, "images"), target_directory_path)
    for item_number, (copied_file_path, image_entry) in enumerate(zip(copied_positive_file_paths, positive_image_annotations)):
        copied_dir_path, copied_file_name = os.path.dirname(copied_file_path), os.path.basename(copied_file_path)
        new_filename = "{data_origin}_{data_number}{file_suffix}".format(
            data_origin=pathlib.Path(positive_data_directory_path).stem,
            data_number=str(item_number).zfill(number_of_leading_zeros),
            file_suffix=pathlib.Path(copied_file_name).suffix
        )
        image_entry["file_name"] = new_filename
        new_file_path = os.path.join(copied_dir_path, new_filename)
        os.rename(copied_file_path, new_file_path)

    print("> Copying negative data directory...")
    copied_negative_file_paths = copy_tree(os.path.join(negative_data_directory_path, "images"), target_directory_path)
    for item_number, (copied_file_path, image_entry) in enumerate(zip(copied_negative_file_paths, negative_image_annotations)):
        copied_dir_path, copied_file_name = os.path.dirname(copied_file_path), os.path.basename(copied_file_path)
        new_filename = "{data_origin}_{data_number}{file_suffix}".format(
            data_origin=pathlib.Path(negative_data_directory_path).stem,
            data_number=str(item_number).zfill(number_of_leading_zeros),
            file_suffix=pathlib.Path(copied_file_name).suffix
        )
        image_entry["file_name"] = new_filename
        new_file_path = os.path.join(copied_dir_path, new_filename)
        os.rename(copied_file_path, new_file_path)

    return positive_image_annotations, negative_image_annotations


def __enforce_class_balance(image_directory_path, coco_annotation):
    print("> Creating class balance...")

    all_categories = [d["category_id"] for d in coco_annotation["annotations"]]
    unique_categories, category_counts = np.unique(all_categories, return_counts=True)

    unique_category_to_prune = np.argmax(category_counts)
    number_to_prune = abs(category_counts[0] - category_counts[1])

    image_ids_to_remove = [
        d["image_id"]
        for d in coco_annotation["annotations"]
        if d["category_id"] == unique_category_to_prune
    ]
    random_engine.shuffle(image_ids_to_remove)
    image_ids_to_remove = random_engine.sample(image_ids_to_remove, number_to_prune)

    image_names_to_remove = [
        d["file_name"] for d in coco_annotation["images"]
        if d["id"] in image_ids_to_remove
    ]

    for image_name in image_names_to_remove:
        os.remove(os.path.join(image_directory_path, image_name))

    coco_annotation["images"] = [
        d for d in coco_annotation["images"]
        if d["id"] not in image_ids_to_remove
    ]

    coco_annotation["annotations"] = [
        d for d in coco_annotation["annotations"]
        if d["image_id"] not in image_ids_to_remove
    ]


def create_merged_annotation(positive_image_annotations, negative_image_annotations):
    print("> Merging annotations together...")

    random_engine.shuffle(positive_image_annotations)
    random_engine.shuffle(negative_image_annotations)

    unique_positive_ids = (np.arange(len(positive_image_annotations)) + 1).tolist()
    unique_negative_ids = (np.arange(len(negative_image_annotations)) + 1 + max(unique_positive_ids)).tolist()

    return {
        "info": create_info(),
        "licenses": create_licenses(),
        "images": create_merged_images(
            unique_positive_ids,
            unique_negative_ids,
            positive_image_annotations,
            negative_image_annotations
        ),
        "categories": create_merged_categories(),
        "annotations": create_merged_annotations(
            unique_positive_ids,
            unique_negative_ids
        ),
    }


def create_merged_images(
        unique_positive_ids,
        unique_negative_ids,
        positive_image_annotations,
        negative_image_annotations
):
    merged_images = [
        {
            "license": None,
            "file_name": dictionary["file_name"],
            "coco_url": None,
            "height": dictionary["height"],
            "width": dictionary["width"],
            "date_captured": None,
            "flickr_url": None,
            "id": unique_id
        }
        for (unique_id, dictionary)
        in zip(unique_positive_ids, positive_image_annotations)
    ]

    merged_images.extend([
        {
            "license": None,
            "file_name": dictionary["file_name"],
            "coco_url": None,
            "height": dictionary["height"],
            "width": dictionary["width"],
            "date_captured": None,
            "flickr_url": None,
            "id": unique_id
        }
        for (unique_id, dictionary)
        in zip(unique_negative_ids, negative_image_annotations)
    ])

    return merged_images


def create_merged_annotations(
        unique_positive_ids,
        unique_negative_ids
):
    merged_annotations = [
        {
            "segmentation": None,
            "area": None,
            "iscrowd": None,
            "image_id": unique_id,
            "bbox": None,
            "category_id": POSITIVE_CATEGORY,
            "id": unique_id
        }
        for unique_id in unique_positive_ids
    ]

    merged_annotations.extend([
        {
            "segmentation": None,
            "area": None,
            "iscrowd": None,
            "image_id": unique_id,
            "bbox": None,
            "category_id": NEGATIVE_CATEGORY,
            "id": unique_id
        }
        for unique_id in unique_negative_ids
    ])

    return merged_annotations


def create_merged_categories():
    merged_categories = [
        {"id": POSITIVE_CATEGORY, "name": "positive_class", "supercategory": "positive_class"},
        {"id": NEGATIVE_CATEGORY, "name": "negative_class", "supercategory": "negative_class"},
    ]

    return merged_categories


def create_info():
    return {
        "description": "Mixed Dataset",
        "url": None,
        "version": "1.0",
        "date_created": f"{date.today().strftime('%d/%m/%Y')}",
    }


def create_licenses():
    return []


def __length_of_number(number):
    return math.ceil(math.log10(number))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("parameters")

    parser.add_argument(
        "--positive_data_directory_path", type=str, help="The path to the positive data directory"
    )
    parser.add_argument(
        "--negative_data_directory_path", type=str, help="The path to the negative data directory"
    )
    parser.add_argument(
        "--testing_fraction", type=float, help="The fraction of the total dataset allocated for testing"
    )
    parser.add_argument(
        "--validation_fraction", type=float, nargs="?", help="The fraction of the training fraction allocated for validation"
    )
    parser.add_argument(
        "--output_folder_name", type=str, help="The path to the mixed dataset"
    )

    print("> Running mixed dataset script...")
    main(arguments=parser.parse_args())
