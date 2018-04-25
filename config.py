import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

TENSOR_BOARD = os.path.join(ROOT_DIR, "tensorboard/")

DATASET = os.path.join(ROOT_DIR, "dataset")
CAL101 = os.path.join(ROOT_DIR, "dataset/caltech101")
CAL101_TRAIN = os.path.join(CAL101, "train")
CAL101_VAL = os.path.join(CAL101, "val")

CAL101_TRAIN_WAVE = os.path.join(DATASET, "wave/train")
CAL101_VAL_WAVE = os.path.join(DATASET, "wave/val")


CAL256 = os.path.join(ROOT_DIR, "dataset/caltech256")
CAL256_TRAIN = os.path.join(CAL256, "train")
CAL256_VAL = os.path.join(CAL256, "val")

CAL256_TRAIN_WAVE = os.path.join(DATASET, "wave256/train")
CAL256_VAL_WAVE = os.path.join(DATASET, "wave256/val")

VALID_EXTS = [".jpg", ".gif", ".png", ".jpeg"]