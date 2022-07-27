import requests
import tarfile
import os
import zipfile
import pathlib
import shutil
import json

from tqdm import tqdm

DATA_DIRECTORY_PATH = "resources/data"


def download_from_url(url, target_path=DATA_DIRECTORY_PATH):
    block_size = 8192

    # Assumes last part of url is a suitable filename
    download_target = os.path.join(target_path, url.split("/")[-1])
    response = requests.get(url, stream=True)
    download_size = int(response.headers.get("content-length", 0))
    progress_bar = tqdm(total=download_size, unit="iB", unit_scale=True)

    print(os.getcwd())

    with open(download_target, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)

    progress_bar.close()

    return download_target


def load_annotation_from_data_path(data_path):
    data_annotation_path = os.path.join(data_path, "annotations\\annotation.json")

    with open(data_annotation_path, "r") as infile:
        annotation = json.load(infile)

    return annotation


def move_downloaded_data(target_data_path, target_annotation_path, extracted_data_path, annotation_file_path):
    shutil.copytree(extracted_data_path, target_data_path, copy_function=shutil.move, dirs_exist_ok=True)
    shutil.move(annotation_file_path, os.path.join(target_annotation_path, "annotation.json"))


def extract_zip_archive(archive_path):
    target_directory_path = os.path.dirname(os.path.realpath(archive_path))
    print("> Extracting zip archive to '{path}'...".format(path=target_directory_path))
    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(path=target_directory_path)

    extracted_folder_name = pathlib.Path(archive_path).stem

    return os.path.join(target_directory_path, extracted_folder_name)


def extract_tar_archive(archive_path, compression_type):
    __validate_compression_type(compression_type)
    mode = f"r:{compression_type}"

    target_directory_path = os.path.dirname(os.path.realpath(archive_path))
    print("> Extracting tar archive to '{path}'...".format(path=target_directory_path))
    with tarfile.open(name=archive_path, mode=mode) as archive:
        archive.extractall(path=target_directory_path)

    return os.path.join(target_directory_path, archive_path.split(".")[0].split("\\")[-1])


def __validate_compression_type(compression_type):
    # gzip, bzip2, lzma:
    valid_compression = ["gz", "bz2", "xz"]
    if compression_type not in valid_compression:
        raise ValueError(
            f"{compression_type} is not a valid compression type.\n The allowed compression types are {valid_compression}"
        )


def get_temporary_data_directory_name():
    return "temporary"


def get_temporary_data_directory_path():
    return os.path.join(DATA_DIRECTORY_PATH, get_temporary_data_directory_name())


def setup_temporary_data_directory():
    directory_path = get_temporary_data_directory_path()

    if not os.path.exists(directory_path):
        os.mkdir(directory_path)

    return directory_path


def setup_data_directory(directory_name):
    directory_path = os.path.join(DATA_DIRECTORY_PATH, directory_name)
    images_path = os.path.join(directory_path, "images")
    annotations_path = os.path.join(directory_path, "annotations")

    if not os.path.exists(directory_path):
        os.mkdir(directory_path)
    if not os.path.exists(images_path):
        os.mkdir(images_path)
    if not os.path.exists(annotations_path):
        os.mkdir(annotations_path)

    return images_path, annotations_path


def is_existing_data_directory(data_directory_name):
    data_path = os.path.join(DATA_DIRECTORY_PATH, data_directory_name)

    return os.path.exists(data_path)
