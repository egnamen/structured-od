import argparse

from sources.scripts import utilities as script_utilities
from sources.model import utilities as model_utilities


def main(arguments):
    config = script_utilities.parse_config_file(arguments.config_file_path)

    dataset = script_utilities.create_dataset(config)
    loader = script_utilities.create_dataloader(config, dataset)

    part_detectors_model = script_utilities.create_part_detectors_model(config, loader)
    model_utilities.save_part_detectors_model(
        file_name=config["run"]["part_detectors_model_file_name"],
        model=part_detectors_model
    )

    # # TODO - Temporary testing code
    # import matplotlib.pyplot as plt
    # import numpy as np
    #
    # input_batch, annotation_batch = next(iter(loader))
    # image_batch = script_utilities.convert_tensor_to_image(input_batch)
    # probability_maps_batch = part_detectors_model(input_batch)
    #
    # batch = 1
    # p_threshold = 0.25
    # _, filter_size, _, _ = probability_maps_batch.shape
    # segmentation_maps_batch = script_utilities.dcn(probability_maps_batch.amax(axis=1)).astype(np.float32)
    # part_hit_maps_batch = script_utilities.dcn(probability_maps_batch.argmax(axis=1)).astype(np.float32)
    # part_hit_maps_batch[segmentation_maps_batch < p_threshold] = np.nan
    #
    # figure, axes = plt.subplots(1, 3, figsize=(18, 4))
    # axes[0].imshow(image_batch[batch])
    # axes[1].imshow(segmentation_maps_batch[batch], vmin=0, vmax=1)
    # cmap = plt.get_cmap('gist_ncar', filter_size + 1).copy()
    # cmap.set_bad(color="k")
    # imshow = axes[2].imshow(part_hit_maps_batch[batch], cmap=cmap, vmin=0, vmax=filter_size + 1)
    # colorbar = figure.colorbar(imshow, ax=axes[2], ticks=np.arange(0, filter_size + 1), drawedges=True)

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
