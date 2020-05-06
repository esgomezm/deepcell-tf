import os
import errno
import numpy as np
import deepcell
from sklearn.model_selection import train_test_split

test_size = 0.1 # % of data saved as test
seed = 0 # seed for random train-test split
filename = 'mousebrain.npz'
DATA_DIR = 'data/'
DATA_FILE = os.path.join(DATA_DIR, filename)

training_data = np.load(DATA_FILE)
X = training_data['X']
y = training_data['y']

filename = 'mousebrain_reduced.npz'
DATA_DIR = 'data/'
DATA_FILE = os.path.join(DATA_DIR, filename)

np.savez(DATA_FILE, X = X[:100], y = y[:100])
print(DATA_FILE)
# confirm the data file is available
assert os.path.isfile(DATA_FILE)


training_data = np.load(DATA_FILE)
X = training_data['X']
y = training_data['y']
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed)
del X, y
print('X.shape: {}\ny.shape: {}'.format(X_train.shape, y_train.shape))
print('X.shape: {}\ny.shape: {}'.format(X_test.shape, y_test.shape))


# Set up other required filepaths

# If the data file is in a subdirectory, mirror it in MODEL_DIR and LOG_DIR
PREFIX = os.path.relpath(os.path.dirname(DATA_FILE), DATA_DIR)

# ROOT_DIR = '/data'  # TODO: Change this! Usually a mounted volume
ROOT_DIR = '3dsamplebased_seg/'
MODEL_DIR = os.path.abspath(os.path.join(ROOT_DIR, 'models', PREFIX))
LOG_DIR = os.path.abspath(os.path.join(ROOT_DIR, 'logs', PREFIX))

# create directories if they do not exist
for d in (MODEL_DIR, LOG_DIR):
    try:
        os.makedirs(d)
    except OSError as exc:  # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise
#-----------------------------------------------------------------------------------------
from tensorflow.keras.optimizers import SGD
from deepcell.utils.train_utils import rate_scheduler

fgbg_model_name = 'sample_fgbg_3d_model'
sample_model_name = 'sample_edgeseg_3d_model'

n_epoch = 2  # Number of training epochs
receptive_field = 61  # should be adjusted for the scale of the data

optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

lr_sched = rate_scheduler(lr=0.01, decay=0.99)

# Transformation settings
transform = 'pixelwise'
dilation_radius = 1  # change dilation radius for edge dilation
separate_edge_classes = True  # break edges into cell-background edge, cell-cell edge
n_features = 4 if separate_edge_classes else 3

# 3D Settings
frames_per_batch = 3
norm_method = 'whole_image'  # data normalization - `whole_image` for 3d conv

# Sample mode settings
batch_size = 8  # number of images per batch (should be 2 ^ n)
win = (receptive_field - 1) // 2  # sample window size
win_z = (frames_per_batch - 1) // 2 # z window size
balance_classes = True  # sample each class equally
max_class_samples = 1e6  # max number of samples per class.

from deepcell import model_zoo

fgbg_model = model_zoo.bn_feature_net_3D(
    receptive_field=receptive_field,
    n_features=2,
    norm_method=norm_method,
    n_frames=frames_per_batch,
    n_channels=X_train.shape[-1])

from deepcell.training import train_model_sample

fgbg_model = train_model_sample(
    model=fgbg_model,
    dataset=DATA_FILE,  # full path to npz file
    model_name=fgbg_model_name,
    test_size=test_size,
    seed=seed,
    window_size=(win, win, (frames_per_batch - 1) // 2),
    optimizer=optimizer,
    batch_size=batch_size,
    balance_classes=balance_classes,
    max_class_samples=max_class_samples,
    transform='fgbg',
    n_epoch=n_epoch,
    model_dir=MODEL_DIR,
    lr_sched=lr_sched,
    rotation_range=180,
    flip=True,
    shear=False)


from deepcell import model_zoo

sample_model = model_zoo.bn_feature_net_3D(
    receptive_field=receptive_field,
    n_features=4,  # (background edge, interior edge, cell interior, background)
    n_frames=frames_per_batch,
    norm_method=norm_method,
    n_channels=X_train.shape[-1])

from deepcell.training import train_model_sample

sample_model = train_model_sample(
    model=sample_model,
    dataset=DATA_FILE,  # full path to npz file
    window_size=(win, win, (frames_per_batch - 1) // 2),
    model_name=sample_model_name,
    test_size=test_size,
    seed=seed,
    transform=transform,
    separate_edge_classes=separate_edge_classes,
    dilation_radius=dilation_radius,
    optimizer=optimizer,
    batch_size=batch_size,
    balance_classes=balance_classes,
    max_class_samples=max_class_samples,
    n_epoch=n_epoch,
    log_dir=LOG_DIR,
    model_dir=MODEL_DIR,
    lr_sched=lr_sched,
    rotation_range=180,
    flip=True,
    shear=False,
    zoom_range=(0.8, 1.2))


fgbg_weights_file = os.path.join(MODEL_DIR, '{}.h5'.format(fgbg_model_name))
fgbg_model.save_weights(fgbg_weights_file)

sample_weights_file = os.path.join(MODEL_DIR, '{}.h5'.format(sample_model_name))
sample_model.save_weights(sample_weights_file)






