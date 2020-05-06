import os
import errno
import numpy as np
import deepcell
from sklearn.model_selection import train_test_split

test_size = 0.1 # % of data saved as test
seed = 0 # seed for random train-test split
filename = 'mousebrain_reduced.npz'
DATA_DIR = 'data/'
DATA_FILE = os.path.join(DATA_DIR, filename)
print(DATA_FILE)
# confirm the data file is available
assert os.path.isfile(DATA_FILE)


training_data = np.load(DATA_FILE)
X = training_data['X']
y = training_data['y']
print(y.shape)
X_train, X_test_total, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed)
del X, y
print('X.shape: {}\ny.shape: {}'.format(X_train.shape, y_train.shape))
print('X.shape: {}\ny.shape: {}'.format(X_test_total.shape, y_test.shape))


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

## Load trained models
from deepcell import model_zoo

# Mask segmentation
fgbg_weights_file = os.path.join(MODEL_DIR, '{}.h5'.format(fgbg_model_name))
run_fgbg_model = model_zoo.bn_feature_net_3D(
    receptive_field=receptive_field,
    dilated=True,
    n_features=2,
    n_frames=frames_per_batch,
    input_shape=tuple(X_test_total.shape[1:]))
run_fgbg_model.load_weights(fgbg_weights_file)

# Edge/interior segmentation
sample_weights_file = os.path.join(MODEL_DIR, '{}.h5'.format(sample_model_name))
run_watershed_model = model_zoo.bn_feature_net_3D(
    receptive_field=receptive_field,
    dilated=True,
    n_features=4,
    n_frames=frames_per_batch,
    input_shape=tuple(X_test_total.shape[1:]))
run_watershed_model.load_weights(sample_weights_file)

## Compute results
PLOT_DIR = os.path.abspath(os.path.join(ROOT_DIR, 'plots', PREFIX))
print(PLOT_DIR)
# create directories if they do not exist
try:
    os.makedirs(PLOT_DIR)
except OSError as exc:  # Guard against race condition
    if exc.errno != errno.EEXIST:
        raise

import matplotlib.pyplot as plt
from skimage.measure import label
from skimage import morphology

for index in range(X_test_total.shape[0]):
  X_test = np.copy(X_test_total[index])
  X_test = X_test.reshape((1, X_test_total.shape[1], X_test_total.shape[2], X_test_total.shape[3], X_test_total.shape[4]))  
  ## Predict images
  test_images = run_watershed_model.predict(X_test)
  print('edge/interior shape:', test_images.shape)
  test_images_fgbg = run_fgbg_model.predict(X_test)
  print('fgbg mask shape:', test_images_fgbg.shape)
  # threshold the foreground/background
  # and remove back ground from edge transform
  threshold = 0.9
  fg_thresh = test_images_fgbg[..., 1] > threshold
  fg_thresh = np.expand_dims(fg_thresh, axis=-1)

  test_images_post_fgbg = test_images * fg_thresh

  # Label interior predictions
  labeled_images = []
  for i in range(test_images_post_fgbg.shape[0]):
      interior = test_images_post_fgbg[i, ..., 2] > .2
      labeled_image = label(interior)
      labeled_image = morphology.remove_small_objects(
          labeled_image, min_size=50, connectivity=1)
      labeled_images.append(labeled_image)
  labeled_images = np.array(labeled_images)
  labeled_images = np.expand_dims(labeled_images, axis=-1)

  print('labeled_images shape:', labeled_images.shape)

  # index = np.random.randint(low=0, high=labeled_images.shape[0])
  frame = np.random.randint(low=0, high=labeled_images.shape[1])
  print('Image:', index)
  print('Frame:', frame)
  fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(15, 15), sharex=True, sharey=True)
  ax = axes.ravel()

  ax[0].imshow(X_test_total[index, frame, ..., 0])
  ax[0].set_title('Source Image')
  
  ax[1].imshow(test_images_fgbg[0, frame, ..., 1])
  ax[1].set_title('Segmentation Prediction')

  ax[2].imshow(fg_thresh[0, frame, ..., 0], cmap='jet')
  ax[2].set_title('FGBG Threshold {}%'.format(threshold * 100))

  ax[3].imshow(test_images[0, frame, ..., 0] + test_images[0, frame, ..., 1], cmap='jet')
  ax[3].set_title('Edge Prediction')

  ax[4].imshow(test_images[0, frame, ..., 2], cmap='jet')
  ax[4].set_title('Interior Prediction')

  ax[5].imshow(labeled_images[0, frame, ..., 0], cmap='jet')
  ax[5].set_title('Instance Segmentation')

  fig.tight_layout()
  fig.savefig(os.path.join(PLOT_DIR, '{}.png'.format(np.str(index))), bbox_inches='tight')
  plt.close('all')

