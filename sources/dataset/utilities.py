import os
import shutil
import json

from sklearn.model_selection import train_test_split

from sources.constant import GLOBAL_SEED

DATASET_DIRECTORY_PATH = "resources/dataset"


def create_partitioned_dataset(
        dataset_name,
        coco_annotation,
        image_directory_path,
        testing_fraction=0.2,
        validation_fraction=None
):
    print("> Creating partitioned dataset...")

    dataset_directory_path = setup_dataset_directory(dataset_name)

    training_annotation, testing_annotation = __partition_annotation(coco_annotation, testing_fraction)
    __setup_partition("testing", dataset_directory_path, testing_annotation, image_directory_path)

    if validation_fraction is not None:
        training_annotation, validation_annotation = __partition_annotation(training_annotation, testing_fraction)
        __setup_partition("validation", dataset_directory_path, validation_annotation, image_directory_path)

    __setup_partition("training", dataset_directory_path, training_annotation, image_directory_path)


def __setup_partition(partition_name, partition_path, annotation, annotation_image_directory_path,):
    root_directory_path = os.path.join(partition_path, partition_name)
    partition_image_directory_path, partition_annotation_directory_path = __create_partition_directories(root_directory_path)

    __move_annotation_images_to_directory(
        annotation=annotation,
        source_directory_path=annotation_image_directory_path,
        target_directory_path=partition_image_directory_path
    )

    __save_annotation_to_directory(
        annotation=annotation,
        target_directory_path=partition_annotation_directory_path
    )


def __save_annotation_to_directory(annotation, target_directory_path):
    annotation_path = os.path.join(target_directory_path, "annotation.json")
    with open(annotation_path, "w") as outfile:
        json.dump(annotation, outfile)


def __move_annotation_images_to_directory(annotation, source_directory_path, target_directory_path):
    for dictionary in annotation["images"]:
        source_image_path = os.path.join(source_directory_path, dictionary["file_name"])
        target_image_path = os.path.join(target_directory_path)
        shutil.move(source_image_path, target_image_path)


def __partition_annotation(annotation, fraction):
    all_images = annotation["images"]
    all_image_ids = [dictionary["id"] for dictionary in all_images]

    all_category_ids = []
    for image_id in all_image_ids:
        image_category_id = [
            dictionary["category_id"]
            for dictionary in annotation["annotations"]
            if dictionary["image_id"] == image_id
        ][0]
        all_category_ids.append(image_category_id)

    first_images, second_images, _, _ = train_test_split(
        all_images,
        all_images,
        test_size=fraction,
        random_state=GLOBAL_SEED,
        stratify=all_category_ids
    )

    first_annotation = annotation.copy()
    first_annotation["images"] = first_images
    first_annotation = __prune_annotations_without_image(first_annotation)

    second_annotation = annotation.copy()
    second_annotation["images"] = second_images
    second_annotation = __prune_annotations_without_image(second_annotation)

    return first_annotation, second_annotation


def __prune_annotations_without_image(annotation):
    pruned_annotation = annotation.copy()

    image_ids = [d["id"] for d in pruned_annotation["images"]]
    pruned_annotation["annotations"] = [d for d in pruned_annotation["annotations"] if d["image_id"] in image_ids]

    return pruned_annotation


def setup_dataset_directory(directory_name):
    directory_path = os.path.join(DATASET_DIRECTORY_PATH, directory_name)

    if not os.path.exists(directory_path):
        os.mkdir(directory_path)

    return directory_path


def is_existing_dataset_directory(dataset_directory_name):
    dataset_path = os.path.join(DATASET_DIRECTORY_PATH, dataset_directory_name)

    return os.path.exists(dataset_path)


def __create_partition_directories(partition_directory_path):
    partition_images_path = os.path.join(partition_directory_path, "images")
    partition_annotations_path = os.path.join(partition_directory_path, "annotations")

    if not os.path.exists(partition_directory_path):
        os.mkdir(partition_directory_path)
    if not os.path.exists(partition_images_path):
        os.mkdir(partition_images_path)
    if not os.path.exists(partition_annotations_path):
        os.mkdir(partition_annotations_path)

    return partition_images_path, partition_annotations_path
