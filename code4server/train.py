import os
import errno
# import numpy as np
# import deepcell
from tensorflow.keras.optimizers import SGD
from deepcell.utils.train_utils import rate_scheduler
from deepcell import model_zoo
# from deepcell.training import train_model_sample

# we need this line as some code from deepcell has been modified
from own_training import train_model_sample_fromdirectory
import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--n_epoch', type=int, default=100, help='Number of epochs to train for')
parser.add_argument('--root_dir', type=str, default="results_training/", help='Name of folder in which all the results and logs will be stored')
parser.add_argument('--dataset', type=str, default="/data/", help='Dataset you are using.') #NEED TO PUT A FULL DIRECTORY, IT WON'T WORK ONLY PUTTING THE FOLDER
parser.add_argument('--receptive_field', type=int, default=512, help='Receptive field for the classification of each pixel')
parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate')
# parser.add_argument('--reduce_lr', type=str2bool, default=True, help='Whether to reduce the learning rate during training')
parser.add_argument('--balance_classes', type=str2bool, default=True, help='Sample each class equally')
parser.add_argument('--max_class_samples', type=int, default=1000, help='Max number of samples per class. #reducir para colab')
args = parser.parse_args()

# confirm the data file is available
assert os.path.isdir(args.dataset)

MODEL_DIR = os.path.abspath(os.path.join(args.root_dir, 'models'))
LOG_DIR = os.path.abspath(os.path.join(args.root_dir, 'logs'))

# create directories if they do not exist
for d in (MODEL_DIR, LOG_DIR):
    try:
        os.makedirs(d)
    except OSError as exc:  # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

fgbg_model_name = 'sample_fgbg_model'
norm_method = 'median'  # data normalization

optimizer = SGD(lr=args.lr, decay=1e-6, momentum=0.9, nesterov=True)
lr_sched = rate_scheduler(lr=args.lr, decay=0.99)

# Sample mode settings
win = (args.receptive_field - 1) // 2  # sample window size

# Transformation settings
transform = 'pixelwise'
dilation_radius = 1  # change dilation radius for edge dilation
separate_edge_classes = True  # break edges into cell-background edge, cell-cell edge
n_features = 4 if separate_edge_classes else 3

fgbg_model = model_zoo.bn_feature_net_2D(
    receptive_field=args.receptive_field,
    n_features=2)
fgbg_model = train_model_sample_fromdirectory(
    model=fgbg_model,
    dataset=args.dataset,  # full path to npz file
    model_name=fgbg_model_name,
    seed=1,
    optimizer=optimizer,
    window_size=(win, win),
    batch_size=args.batch_size,
    transform='fgbg',
    n_epoch=args.n_epoch,
    balance_classes=args.balance_classes,
    max_class_samples=args.max_class_samples,
    model_dir=MODEL_DIR,
    log_dir=LOG_DIR,
    lr_sched=lr_sched,
    rotation_range=180,
    flip=True,
    shear=False,
    zoom_range=(0.8, 1.2))


