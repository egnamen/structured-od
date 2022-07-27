import os
import argparse

from sources.scripts import utilities as script_utilities
from sources.model import utilities as model_utilities


def main(arguments):
    config = script_utilities.parse_config_file(arguments.config_file_path)

    dataset = script_utilities.create_dataset(config)
    loader = script_utilities.create_dataloader(config, dataset)

    part_detectors_model = script_utilities.create_part_detectors_model(config, loader)

    output_file_path = os.path.join(model_utilities.get_model_folder_path(), config["model"]["output_name"])
    model_utilities.save_part_detectors_model(output_file_path, part_detectors_model)

    # # TODO - Temporary testing code
    # import matplotlib.pyplot as plt
    #
    # input_batch, annotation_batch = next(iter(loader))
    # probability_maps_batch = part_detectors_model(input_batch)
    #
    # probability_maps_batch = script_utilities.dcn(probability_maps_batch)
    # image_batch = script_utilities.convert_tensor_to_image(input_batch)
    #
    # batch, filter = 0, 0
    # image = image_batch[batch]
    # probability_maps = probability_maps_batch[batch]
    #
    # figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    # axes[0].imshow(image)
    # imshow = axes[1].imshow(probability_maps[filter], vmin=0.00, vmax=1.00)
    # figure.colorbar(imshow, ax=axes[1])
    # imshow = axes[2].imshow(probability_maps.max(axis=0), vmin=0.00, vmax=1.00)
    # figure.colorbar(imshow, ax=axes[2])
    # plt.gcf().set_tight_layout(True)

    exit(0)


if __name__ == "__main__":
    print("> Running main of script...")

    parser = argparse.ArgumentParser("parameters")
    parser.add_argument(
        "--config_file_path",
        required=True,
        type=str,
        help="The config file to use for running script",
    )

    main(arguments=parser.parse_args())
