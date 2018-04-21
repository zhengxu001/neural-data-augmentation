import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

TENSOR_BOARD = os.path.join(ROOT_DIR, "tensorboard/")

CAL101 = os.path.join(ROOT_DIR, "dataset/caltech101")
CAL101_TRAIN = os.path.join(CAL101, "train")
CAL101_VAL = os.path.join(CAL101, "val")

CAL101 = os.path.join(ROOT_DIR, "dataset/caltech101")
CAL101_TRAIN_STYLE1 = os.path.join(CAL101, "train_style1")
CAL101_VAL_STYLE1 = os.path.join(CAL101, "val_style1")


VALID_EXTS = [".jpg", ".gif", ".png", ".jpeg"]