import os
import errno
import numpy as np
import deepcell


# the path to the data file is currently required for `train_model_()` functions
filename = 'example.npz'
test_size = 0.1 # % of data saved as test
seed = 0 # seed for random train-test split
DATA_DIR = './data/'
# DATA_DIR = '/home/esgomezm/Documents/3D-PROTUCEL/data/small_data/'
# DATA_FILE should be a npz file, preferably from `make_training_data`
DATA_FILE = os.path.join(DATA_DIR, filename)
# confirm the data file is available
assert os.path.isfile(DATA_FILE)
# DATA_FILE = DATA_DIR
# assert os.path.isdir(DATA_FILE)
# Set up other required filepaths

# If the data file is in a subdirectory, mirror it in MODEL_DIR and LOG_DIR
PREFIX = os.path.relpath(os.path.dirname(DATA_FILE), DATA_DIR)

ROOT_DIR = './2dsamplebased_seg'
MODEL_DIR = os.path.abspath(os.path.join(ROOT_DIR, 'models', PREFIX))
LOG_DIR = os.path.abspath(os.path.join(ROOT_DIR, 'logs', PREFIX))

# create directories if they do not exist
for d in (MODEL_DIR, LOG_DIR):
    try:
        os.makedirs(d)
    except OSError as exc:  # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

from tensorflow.keras.optimizers import SGD
from deepcell.utils.train_utils import rate_scheduler

fgbg_model_name = 'sample_fgbg_model'
n_epoch = 1  # Number of training epochs
norm_method = 'median'  # data normalization
receptive_field = 61  # should be adjusted for the scale of the data

optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

lr_sched = rate_scheduler(lr=0.01, decay=0.99)

# Sample mode settings
batch_size = 1 # 64  # number of images per batch (should be 2 ^ n)
win = (receptive_field - 1) // 2  # sample window size
balance_classes = False  # sample each class equally
max_class_samples = 50  # max number of samples per class.

# Transformation settings
transform = 'pixelwise'
dilation_radius = 1  # change dilation radius for edge dilation
separate_edge_classes = True  # break edges into cell-background edge, cell-cell edge
n_features = 4 if separate_edge_classes else 3


from deepcell import model_zoo

fgbg_model = model_zoo.bn_feature_net_2D(
    receptive_field=receptive_field,
    n_features=2)

from deepcell.training import train_model_sample

fgbg_model = train_model_sample(
    model=fgbg_model,
    dataset=DATA_FILE,  # full path to npz file
    model_name=fgbg_model_name,
    test_size=0.5,
    seed=seed,
    optimizer=optimizer,
    window_size=(win, win),
    batch_size=batch_size,
    transform='fgbg',
    n_epoch=n_epoch,
    balance_classes=balance_classes,
    max_class_samples=max_class_samples,
    model_dir=MODEL_DIR,
    log_dir=LOG_DIR,
    lr_sched=lr_sched,
    rotation_range=180,
    flip=True,
    shear=False,
    zoom_range=(0.8, 1.2))
