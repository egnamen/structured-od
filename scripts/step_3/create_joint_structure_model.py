import argparse

from sources.scripts import utilities as script_utilities
from sources.model import utilities as model_utilities


def main(arguments):
    config = script_utilities.parse_config_file(arguments.config_file_path)

    dataset = script_utilities.create_dataset(config)
    loader = script_utilities.create_dataloader(config, dataset)

    part_detectors_model = model_utilities.load_part_detectors_model(config["run"]["part_detectors_model_file_name"])
    joint_structure_model = script_utilities.create_joint_structure_model(config, part_detectors_model)

    joint_structure_model = script_utilities.train_joint_structure_model(config, joint_structure_model, loader)

    model_utilities.save_joint_structure_model(
        file_name=config["run"]["joint_structure_model_file_name"],
        model=joint_structure_model
    )

    # # TODO - Temporary testing code
    # # NB: Modified "_export.py" inside "tree.plot_tree(...)" to ensure '>' symbol is used
    #
    # from sklearn import tree
    # import numpy as np
    # import matplotlib.pyplot as plt
    #
    # tree.plot_tree(
    #     joint_structure_model.combination_model,
    #     feature_names=[f"pd [{number}]" for number in np.arange(0, len(part_detectors_model.part_detection_heads)).tolist()],
    #     class_names=["not-car", "car"],
    #     filled=True,
    #     fontsize=6,
    #     label="root",
    #     impurity=False,
    #     node_ids=False,
    #     rounded=True,
    #     precision=0
    # )
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

    # # TODO - Temporary testing code
    # import matplotlib.pyplot as plt
    # import numpy as np
    #
    # input_batch, annotation_batch = next(iter(loader))
    # image_batch = script_utilities.convert_tensor_to_image(input_batch)
    # probability_maps_batch = joint_structure_model.part_detectors_model(input_batch)
    # histogram_batch = joint_structure_model.preprocess_part_detections(probability_maps_batch)
    #
    # batch = 0
    #
    # _, filter_size, _, _ = probability_maps_batch.shape
    # segmentation_maps_batch = script_utilities.dcn(probability_maps_batch.amax(axis=1)).astype(np.float32)
    # part_hit_maps_batch = script_utilities.dcn(probability_maps_batch.argmax(axis=1)).astype(np.float32)
    # part_hit_maps_batch[segmentation_maps_batch < joint_structure_model.probability_threshold] = np.nan
    #
    # figure, axes = plt.subplots(1, 3, figsize=(18, 4))
    # plt.title(f"Prediction = {joint_structure_model(input_batch)[batch]}")
    # axes[0].imshow(image_batch[batch])
    # axes[1].imshow(segmentation_maps_batch[batch], vmin=0, vmax=1)
    # cmap = plt.get_cmap('gist_ncar', filter_size + 1).copy()
    # cmap.set_bad(color="k")
    # imshow = axes[2].imshow(part_hit_maps_batch[batch], cmap=cmap, vmin=0, vmax=filter_size + 1)
    # colorbar = figure.colorbar(imshow, ax=axes[2], ticks=np.arange(0, filter_size + 1), drawedges=True)
    #
    # plt.figure()
    # plt.bar(x=np.arange(0, filter_size).tolist(), height=histogram_batch[batch])
    # for y in np.arange(int(plt.ylim()[0]), int(plt.ylim()[1])):
    #     plt.axhline(y=y, xmin=plt.xlim()[0], xmax=plt.xlim()[1], color="red")
    # xticks = plt.xticks(np.arange(0, filter_size).tolist(), rotation=-90)

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
