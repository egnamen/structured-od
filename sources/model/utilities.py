import os
import torch


def get_part_detectors_folder_path():
    return "resources/model/part_detectors"


def get_joint_structure_folder_path():
    return "resources/model/joint_structure"


def save_part_detectors_model(file_name, model):
    print("> Saving part detectors model...")
    file_path = os.path.join(get_part_detectors_folder_path(), file_name)
    torch.save(model, file_path)


def load_part_detectors_model(file_name):
    print("> Loading part detectors model...")
    file_path = os.path.join(get_part_detectors_folder_path(), file_name)
    return torch.load(file_path)


def save_joint_structure_model(file_name, model):
    print("> Saving joint structure model...")
    file_path = os.path.join(get_joint_structure_folder_path(), file_name)
    torch.save(model, file_path)


def load_joint_structure_model(file_name):
    print("> Loading joint structure model...")
    file_path = os.path.join(get_joint_structure_folder_path(), file_name)
    return torch.load(file_path)
