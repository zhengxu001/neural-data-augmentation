import os
import shutil
import itertools
import numpy as np
import config
import argparse
def main(dataset_name):
    data_path = os.path.join(config.DATASET, dataset_name, "train")
    test_path = os.path.join(config.DATASET, dataset_name, "val")
    categories = os.listdir(data_path)
    for cat in categories:
        image_files = os.listdir(os.path.join(data_path, cat))
        choices = np.random.choice([0, 1], size=(len(image_files),), p=[0.70, .30])
        files_to_move = itertools.compress(image_files, choices)
        for _f in files_to_move:
            origin_path = os.path.join(data_path, cat, _f)
            dest_dir = os.path.join(test_path, cat)
            dest_path = os.path.join(test_path, cat, _f)
            if not os.path.isdir(dest_dir):
                os.mkdir(dest_dir)
            shutil.move(origin_path, dest_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str,
                        default='caltech101',
                        help='Name of this training run.')
    args, unparsed = parser.parse_known_args()
    main(args.dataset_name)