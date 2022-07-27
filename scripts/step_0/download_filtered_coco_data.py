import argparse
import os
import shutil
import json

from sources.data import utilities as data_utilities

FILTERED_COCO_DATA_DIRECTORY = "filtered_coco"


def main(arguments):
    if data_utilities.is_existing_data_directory(FILTERED_COCO_DATA_DIRECTORY):
        raise ValueError("Data directory called '{name}' already exists".format(name=FILTERED_COCO_DATA_DIRECTORY))

    temporary_directory_path = data_utilities.setup_temporary_data_directory()

    # # Debugging code to skip downloading again
    # downloaded_data_path = "resources\\data\\temporary\\val2017.zip"
    # downloaded_annotation_path = "resources\\data\\temporary\\annotations_trainval2017.zip"

    downloaded_data_path, downloaded_annotation_path = download_coco_dataset(temporary_directory_path)
    extracted_data_path, coco_annotation_path = extract_data(downloaded_data_path, downloaded_annotation_path)

    # # Debugging code to skip extracting again
    # temporary_directory_path = "resources\\data\\temporary"
    # extracted_data_path = "resources\\data\\temporary\\val2017"
    # coco_annotation_path = "resources\\data\\temporary\\annotations\\instances_val2017.json"

    if arguments.category_ids_to_remove is not None:
        __prune_on_category_ids(
            category_ids_to_remove=arguments.category_ids_to_remove,
            image_directory_path=extracted_data_path,
            annotation_file_path=coco_annotation_path
        )

    target_data_path, target_annotation_path = data_utilities.setup_data_directory(FILTERED_COCO_DATA_DIRECTORY)
    data_utilities.move_downloaded_data(target_data_path, target_annotation_path, extracted_data_path, coco_annotation_path)

    shutil.rmtree(temporary_directory_path)


def download_coco_dataset(download_directory_path):
    data_URL = "http://images.cocodataset.org/zips/val2017.zip"
    annotation_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

    data_path = data_utilities.download_from_url(data_URL, download_directory_path)
    annotation_path = data_utilities.download_from_url(annotation_URL, download_directory_path)

    return data_path, annotation_path


def extract_data(data_path, annotation_path):
    data_utilities.extract_zip_archive(data_path)
    data_utilities.extract_zip_archive(annotation_path)

    extracted_data_path = os.path.join(os.path.dirname(os.path.realpath(data_path)), "val2017")
    coco_annotation_path = os.path.join(os.path.dirname(os.path.realpath(annotation_path)), "annotations\\instances_val2017.json")

    return extracted_data_path, coco_annotation_path


def __prune_on_category_ids(category_ids_to_remove, image_directory_path, annotation_file_path):
    with open(annotation_file_path, "r") as infile:
        annotation = json.load(infile)

    image_ids_to_remove = []
    annotations_to_keep = []
    for dictionary in annotation["annotations"]:
        if dictionary["category_id"] in category_ids_to_remove:
            image_ids_to_remove.append(dictionary["image_id"])
        else:
            annotations_to_keep.append(dictionary)

    images_to_keep = []
    for dictionary in annotation["images"]:
        if dictionary["id"] in image_ids_to_remove:
            image_file_path = os.path.join(image_directory_path, dictionary["file_name"])
            os.remove(image_file_path)
        else:
            images_to_keep.append(dictionary)

    annotation["images"] = images_to_keep
    annotation["annotations"] = annotations_to_keep

    with open(annotation_file_path, "w") as outfile:
        json.dump(annotation, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("parameters")

    parser.add_argument(
        "--category_ids_to_remove",
        nargs="+",
        type=int,
        help="The category ids of the data that will be pruned"
    )

    print("> Running mixed dataset script...")
    main(arguments=parser.parse_args())
