from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import datetime
import os

import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import callbacks
from tensorflow.python.keras.optimizers import SGD
from deepcell import losses
from deepcell import image_generators
from deepcell.utils import train_utils
from deepcell.utils.train_utils import rate_scheduler
from deepcell.utils.train_utils import get_callbacks



# TODO: make this function compatible with 3D data or images with more than 1 channel.
def random_crop(image, label, crop_height, crop_width):
    """"
    Crop a random patch of size (crop_height, crop_width) in a 2D image following a sampling distribution where patches
    where label>0 have higher probability.
        image: 2D numpy array
        label: 2D numpy array with the segmentation, detection or any information about image
        crop height / crop width: determine the size of the patch to crop.
    Returns:
        patch of the image and the corresponding patch of the label.
    """
    if (image.shape[0] != label.shape[0]) or (image.shape[1] != label.shape[1]):
        raise Exception('Image and label must have the same dimensions!')
    if (crop_width <= image.shape[1]) and (crop_height <= image.shape[0]):
        pdf_im = np.ones(label.shape) # label is a mask
        pdf_im[label > 0] = 10000 # the weight we want to give to positive values in labels.
        pdf_im = pdf_im[:-crop_height,:-crop_width] # limit the coordinates in which a centroid can lay
        prob = np.float32(pdf_im)
        # convert the 2D matrix into a vector and normalize it so you create a distribution of all the possible values
        # between 1 and prod(pdf.shape)(sum=1)
        prob = prob.ravel()/np.sum(prob)
        choices = np.prod(pdf_im.shape)
        # get a random centroid but following a pdf distribution.
        index = np.random.choice(choices, size=1, p=prob)
        coordinates = np.unravel_index(index, shape=pdf_im.shape)
        y = coordinates[0][0]
        x = coordinates[1][0]
        return image[y:y+crop_height, x:x+crop_width], label[y:y+crop_height, x:x+crop_width]
    else:
        raise Exception('Crop shape ({0}, {1}) exceeds image dimensions ({2}, {3})!'.format(crop_height, crop_width, image.shape[0], image.shape[1]))




def load_data_pairs(DATAPATH, mode = 'train', patch_crop=False, crop_height=256, crop_width=256):
    import cv2
    import sys
    files = os.listdir(os.path.join(DATAPATH, mode))
    # files = os.listdir(os.path.join(DATAPATH, mode, 'inputs'))
    X = None
    sys.stdout.write("\rLoading data...\n")
    i = 0
    for fname in files:
        i = i+1
        text = "\r{0} {1}%".format("|" * 20, i/len(files) * 100)
        sys.stdout.write(text)
        sys.stdout.flush()
        input_im = cv2.imread(os.path.join(DATAPATH, mode, fname), cv2.IMREAD_ANYDEPTH)
        # input_im = cv2.imread(os.path.join(DATAPATH, mode, 'inputs', fname), cv2.IMREAD_ANYDEPTH)
        # input_im = input_im[:,:,0]
        mask_im = cv2.imread(os.path.join(DATAPATH, mode + '_labels', 'instance_ids_' + fname[4:]), cv2.IMREAD_ANYDEPTH)
        # mask_im = cv2.imread(os.path.join(DATAPATH, mode+, 'labels', 'instance_ids_' + fname[4:]), cv2.IMREAD_ANYDEPTH)
        # mask_im = mask_im[:,:,0]
        # mask_im[mask_im > 0] = 1

        if patch_crop==True:
            input_im, mask_im = random_crop(input_im, mask_im, crop_height, crop_width)
            input_im = input_im.reshape((1, crop_height, crop_width, 1))
            mask_im = mask_im.reshape((1, crop_height, crop_width, 1))
        else:
            input_im = input_im.reshape((1, input_im.shape[0], input_im.shape[1], 1))
            mask_im = mask_im.reshape((1, mask_im.shape[0], mask_im.shape[1], 1))

        if X is None:
            X = input_im
            y = mask_im
        else:
            X = np.concatenate((X,input_im), axis=0)
            y = np.concatenate((y,mask_im), axis=0)
    return X, y



# TODO: modify this for 3D data or images with more than one channel.
def get_data_from_path(DATAPATH, patch_crop=False, crop_height=256, crop_width=256):
    """
    Read the training, and test 2D data and save them as dictionaries used during the training.
    Args:
        DATAPATH: Main path where the data is stored as train, train_labels, test, test_labels
        patch_crop: Whether we want to crop a small patch of each image
        crop_height: Height size (Y-axis) of the path to crop
        crop_width: Width size (X-axis) of the path to crop

    Returns:
        train_dict with the input and output images. The length of the training data is equal to the number of images
        available in the directory.
        test_dict with the input and output images that belong to the test set. The length of the data is equal to the
        number of images available in the directory. The size of these images could be the original one instead of
        patches, as long as all have the same size.
    """
    X_train, y_train = load_data_pairs(DATAPATH, mode='train', patch_crop=patch_crop,
                                       crop_height=crop_height, crop_width=crop_width)
    X_test, y_test = load_data_pairs(DATAPATH, mode='test', patch_crop=False)
    train_dict = {
        'X': X_train,
        'y': y_train
    }
    test_dict = {
        'X': X_test,
        'y': y_test
    }
    return train_dict, test_dict


def train_model_sample_fromdirectory(model,
                       dataset,
                       expt='',
                       test_size=.2,
                       n_epoch=10,
                       batch_size=32,
                       num_gpus=None,
                       transform=None,
                       window_size=None,
                       balance_classes=True,
                       max_class_samples=None,
                       log_dir='/data/tensorboard_logs',
                       model_dir='/data/models',
                       model_name=None,
                       focal=False,
                       gamma=0.5,
                       optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
                       lr_sched=rate_scheduler(lr=0.01, decay=0.95),
                       rotation_range=0,
                       flip=False,
                       shear=0,
                       zoom_range=0,
                       seed=0,
                       **kwargs):
    """Train a model using sample mode.

    Args:
        model (tensorflow.keras.Model): The model to train.
        dataset (str): Path to a dataset to train the model with.
        expt (str): Experiment, substring to include in model name.
        test_size (float): Percent of data to leave as test data.
        n_epoch (int): Number of training epochs.
        batch_size (int): Number of batches per training step.
        num_gpus (int): The number of GPUs to train on.
        transform (str): Defines the transformation of the training data.
            One of 'watershed', 'fgbg', 'pixelwise'.
        window_size (tuple(int, int)): Size of sampling window
        balance_classes (bool): Whether to perform class-balancing on data
        max_class_samples (int): Maximum number of examples per class to sample
        log_dir (str): Filepath to save tensorboard logs. If None, disables
            the tensorboard callback.
        model_dir (str): Directory to save the model file.
        model_name (str): Name of the model (and name of output file).
        focal (bool): If true, uses focal loss.
        gamma (float): Parameter for focal loss
        optimizer (object): Pre-initialized optimizer object (SGD, Adam, etc.)
        lr_sched (function): Learning rate schedular function
        rotation_range (int): Maximum rotation range for image augmentation
        flip (bool): Enables horizontal and vertical flipping for augmentation
        shear (int): Maximum rotation range for image augmentation
        zoom_range (tuple): Minimum and maximum zoom values (0.8, 1.2)
        seed (int): Random seed
        kwargs (dict): Other parameters to pass to _transform_masks

    Returns:
        tensorflow.keras.Model: The trained model
    """
    is_channels_first = K.image_data_format() == 'channels_first'

    if model_name is None:
        todays_date = datetime.datetime.now().strftime('%Y-%m-%d')
        data_name = os.path.splitext(os.path.basename(dataset))[0]
        model_name = '{}_{}_{}'.format(todays_date, data_name, expt)
    model_path = os.path.join(model_dir, '{}.h5'.format(model_name))
    loss_path = os.path.join(model_dir, '{}.npz'.format(model_name))

    # train_dict, test_dict = get_data(dataset, test_size=test_size, seed=seed)

    train_dict, test_dict = get_data_from_path(dataset, patch_crop=False)
    n_classes = model.layers[-1].output_shape[1 if is_channels_first else -1]

    # the data, shuffled and split between train and test sets
    print('X_train shape:', train_dict['X'].shape)
    print('y_train shape:', train_dict['y'].shape)
    print('X_test shape:', test_dict['X'].shape)
    print('y_test shape:', test_dict['y'].shape)
    print('Output Shape:', model.layers[-1].output_shape)
    print('Number of Classes:', n_classes)

    def loss_function(y_true, y_pred):
        if isinstance(transform, str) and transform.lower() == 'disc':
            return losses.discriminative_instance_loss(y_true, y_pred)
        if focal:
            return losses.weighted_focal_loss(
                y_true, y_pred, gamma=gamma, n_classes=n_classes)
        return losses.weighted_categorical_crossentropy(
            y_true, y_pred, n_classes=n_classes)

    if num_gpus is None:
        num_gpus = train_utils.count_gpus()

    if num_gpus >= 2:
        batch_size = batch_size * num_gpus
        model = train_utils.MultiGpuModel(model, num_gpus)

    print('Training on {} GPUs'.format(num_gpus))

    model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])

    if train_dict['X'].ndim == 4:
        DataGenerator = image_generators.SampleDataGenerator
        window_size = window_size if window_size else (30, 30)
    elif train_dict['X'].ndim == 5:
        DataGenerator = image_generators.SampleMovieDataGenerator
        window_size = window_size if window_size else (30, 30, 3)
    else:
        raise ValueError('Expected `X` to have ndim 4 or 5. Got',
                         train_dict['X'].ndim)

    # this will do preprocessing and realtime data augmentation
    datagen = DataGenerator(
        rotation_range=rotation_range,
        shear_range=shear,
        zoom_range=zoom_range,
        horizontal_flip=flip,
        vertical_flip=flip)

    # no validation augmentation
    datagen_val = DataGenerator(
        rotation_range=0,
        shear_range=0,
        zoom_range=0,
        horizontal_flip=0,
        vertical_flip=0)

    train_data = datagen.flow(
        train_dict,
        seed=seed,
        batch_size=batch_size,
        transform=transform,
        transform_kwargs=kwargs,
        window_size=window_size,
        balance_classes=balance_classes,
        max_class_samples=max_class_samples)
        # save_to_dir= './training_data',
        # save_prefix='t',
        # save_format='tif')

    val_data = datagen_val.flow(
        test_dict,
        seed=seed,
        batch_size=batch_size,
        transform=transform,
        transform_kwargs=kwargs,
        window_size=window_size,
        balance_classes=False,
        max_class_samples=max_class_samples)

    train_callbacks = get_callbacks(
        model_path, lr_sched=lr_sched,
        tensorboard_log_dir=log_dir,
        save_weights_only=num_gpus >= 2,
        monitor='val_loss', verbose=1)

    # fit the model on the batches generated by datagen.flow()
    loss_history = model.fit_generator(
        train_data,
        steps_per_epoch=train_data.y.shape[0] // batch_size,
        epochs=n_epoch,
        validation_data=val_data,
        validation_steps=val_data.y.shape[0] // batch_size,
        callbacks=train_callbacks)

    np.savez(loss_path, loss_history=loss_history.history)

    return model
