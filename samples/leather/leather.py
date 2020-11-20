"""
Mask R-CNN
Train on the Leather dataset.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 leather.py train --dataset=/path/to/leather/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 leather.py train --dataset=/path/to/leather/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 leather.py train --dataset=/path/to/leather/dataset --weights=imagenet
"""

from mrcnn import model as modellib, utils
from mrcnn.config import Config
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class LeatherConfig(Config):
    """Configuration for training on the dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "leather"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 5  # Background + number of defaults

    # Number of training steps per epoch
    # STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    # DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class LeatherDataset(utils.Dataset):

    def load_leather(self, dataset_dir, subset):
        """Load a subset of the leather dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes.
        self.add_class("leather", 1, "color")
        self.add_class("leather", 2, "cut")
        self.add_class("leather", 3, "fold")
        self.add_class("leather", 4, "glue")
        self.add_class("leather", 5, "poke")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir_sub = os.path.join(dataset_dir, subset)

        folders = ["color", "cut", "fold", "glue", "poke"]

        for label in folders:
            label_dir = os.path.join(dataset_dir_sub, label)
            image_ids = next(os.walk(label_dir))[2]

            for img in image_ids:
                path = os.path.join(dataset_dir_sub, label, img)
                image = skimage.io.imread(path)
                height, width = image.shape[:2]
                image_id = "{}_{}".format(label, img)
                if label != "good":
                    mask_dir = os.path.join(
                        dataset_dir, "ground_truth", label,
                        "{}_mask.png".format(img.split('.')[0]))
                else:
                    mask_dir = None
                self.add_image(
                    "leather",
                    image_id=image_id,
                    path=path,
                    width=width, height=height,
                    label=label,
                    mask_dir=mask_dir
                )

    def load_mask(self, image_id):
        """Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        info = self.image_info[image_id]
        class_id = self.class_names.index(info["label"])
        mask_dir = info["mask_dir"]
        mask = np.zeros([info['height'], info['width'], 1], dtype=np.uint8)
        if info["label"] != "good":
            mask_img = skimage.io.imread(mask_dir)
            for i, row in enumerate(mask_img):
                for j, item in enumerate(row):
                    mask[i, j] = [item]

        return mask.astype(np.bool), np.array([class_id]).astype(np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "leather":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)



def train(model):
    # Training dataset.
    dataset_train = LeatherDataset()
    dataset_train.load_leather(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = LeatherDataset()
    dataset_val.load_leather(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')

############################################################
#  Training
############################################################

"""
if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect leathers.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/leather/dataset/",
                        help='Directory of the leather dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = LeatherConfig()
    else:
        class InferenceConfig(leatherConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    else:
        print("'{}' is not recognized. "
              "Use 'train'".format(args.command))

"""