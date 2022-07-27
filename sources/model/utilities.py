import torch


def get_model_folder_path():
    return "resources/model"


def save_part_detectors_model(file_path, model):
    print("> Saving part detectors model...")
    torch.save(model, file_path)


def load_part_detectors_model(file_path, model):
    print("> Load part detectors model...")
    torch.load(model, file_path)
