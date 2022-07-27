import scipy.io
import json
import os
import shutil

from datetime import date
from PIL import Image

from sources.data import utilities as data_utilities

STANFORD_CARS_DATA_DIRECTORY = "stanford_cars"


def main():
    if data_utilities.is_existing_data_directory(STANFORD_CARS_DATA_DIRECTORY):
        raise ValueError("Data directory called '{name}' already exists".format(name=STANFORD_CARS_DATA_DIRECTORY))

    temporary_directory_path = data_utilities.setup_temporary_data_directory()

    downloaded_data_path, downloaded_annotation_path = download_standford_car_dataset(temporary_directory_path)
    extracted_data_path, coco_annotation_path = extract_data(downloaded_data_path, downloaded_annotation_path)

    # # Debugging code to skip extracting again
    # temporary_directory_path = "resources\\data\\temporary"
    # extracted_data_path = "resources\\data\\temporary\\car_ims"
    # coco_annotation_path = "resources\\data\\temporary\\annotations.json"

    target_data_path, target_annotation_path = data_utilities.setup_data_directory(STANFORD_CARS_DATA_DIRECTORY)
    data_utilities.move_downloaded_data(target_data_path, target_annotation_path, extracted_data_path, coco_annotation_path)

    shutil.rmtree(temporary_directory_path)


def download_standford_car_dataset(download_directory_path):
    data_URL = "http://ai.stanford.edu/~jkrause/car196/car_ims.tgz"
    annotation_URL = "http://ai.stanford.edu/~jkrause/car196/cars_annos.mat"

    data_path = data_utilities.download_from_url(data_URL, download_directory_path)
    annotation_path = data_utilities.download_from_url(annotation_URL, download_directory_path)

    return data_path, annotation_path


def extract_data(data_path, annotation_path):
    extracted_data_path = data_utilities.extract_tar_archive(data_path, compression_type="gz")
    coco_annotation_path = mat_to_coco(annotation_path, extracted_data_path)

    return extracted_data_path, coco_annotation_path


def mat_to_coco(annotation_path, data_path):
    print("> Converting annotations from MAT format to COCO format...")
    mat_annotations = scipy.io.loadmat(annotation_path)

    target_directory_path = os.path.dirname(os.path.realpath(annotation_path))
    target_annotation_path = os.path.join(target_directory_path, "annotations.json")
    with open(target_annotation_path, "w") as outfile:
        json.dump(
            {
                "info": create_info(),
                "licenses": create_licenses(),
                "images": create_images(data_path),
                "categories": create_categories(mat_annotations),
                "annotations": create_annotations(mat_annotations),
            },
            outfile,
        )

    return target_annotation_path


def create_info():
    return {
        "description": "COCO variant of the Stanford Car dataset",
        "url": "https://ai.stanford.edu/~jkrause/cars/car_dataset.html",
        "version": "1.0",
        "date_created": f"{date.today().strftime('%d/%m/%Y')}",
    }


def create_licenses():
    # Will just keep it here for now although I'm pretty sure we won't need it...
    return []


def create_images(data_path):
    images = []
    contained_images = os.listdir(data_path)

    for image_name in contained_images:
        with Image.open(os.path.join(data_path, image_name)) as image:
            width, height = image.size
            images.append(
                {
                    "license": -1,
                    "file_name": image_name,
                    "coco_url": "N.A.",
                    "height": height,
                    "width": width,
                    "date_captured": "N.A.",
                    "flickr_url": "N.A.",
                    "id": int(image_name.split(".")[0]),
                }
            )

    return images


def create_categories(mat_annotations):
    categories = []
    supercategoy = "car"
    class_names = mat_annotations["class_names"][0]

    for index, category in enumerate(class_names):
        # Categories begin at 1 and not 0 in the dataset
        category_id = index + 1

        categories.append(
            {
                "id": category_id,
                "name": category[0],
                "supercategory": supercategoy,
            }
        )

    return categories


def create_annotations(mat_annotations):
    annotations = []
    mat_annotations = mat_annotations["annotations"][0]

    for index, annotation in enumerate(mat_annotations):
        # uint18 etc. is not serializable, therefore, cast to int:
        image_id = int("".join(character for character in annotation[0][0] if character.isdigit()))
        left = int(annotation[1][0][0])
        top = int(annotation[2][0][0])
        right = int(annotation[3][0][0])
        bottom = int(annotation[4][0][0])
        width = int(right - left)
        height = int(bottom - top)
        area = width * height

        annotations.append(
            {
                "segmentation": {},
                "area": area,
                "iscrowd": 0,
                "image_id": image_id,
                "bbox": [left, top, width, height],
                "category_id": int(annotation[5][0][0]),
                "id": index,
            }
        )

    return annotations


if __name__ == "__main__":
    main()
