'''
This is used to train/test code for our eye tracking models, as we might end up with one or more
types of models. So this file will support the different types as they are added into the project.

Team: QUBMenInVens
Styling: https://github.com/python/peps/blob/master/pep-0257.txt

'''
import os
import sys
from argparse import ArgumentParser
from EyeTrack.cnn_regression import Cnn_regression
import numpy as np
from EyeTrack.grid_classification_model import Gccn_classification
from EyeTrack.val_to_grid import val_to_grid


def main(from_app=[]):
    """
        Process the user input and set up the correct model and pass in the data or use a pre-trained model for processing data.

        kwargs:
            type    : Model type (At moment only CNN supported)
            train   : If the program should be in train mode.
            data:   : Location of the image data.
            verbose : Allow debug statments to be printed to the console.
        Raises:
            ValueError : If name is not only letter, if data arch isn't supported, or modelLoc is missing when using retrain.
    """
    settings = process_args(from_app)

    if settings.verbose:
        print('Verbose mode is on.')

    data_archs = ['eyeq', 'mit']

    if not settings.name.isalpha():
        raise ValueError('Name can only be letters do not pass in numbers or symbols')

    if settings.data_arch not in data_archs:
        raise ValueError('Data architecture must be either {}'.format(data_archs))

    if not settings.modelLoc and settings.retrain:
        raise ValueError('Use of Retrain flag requires that a model location is passed in please use --modelLoc')

    if settings.train:
        train(settings)
    elif settings.test:
        return test(settings)
    else:
        return 'Invalid input'


def process_args(args_data):
    """
        This method allows the processing of what the user has inputed and check that everything that is required is passed in
        and return the argmenets so that we can have a user setting per script ran.

        Return:  argparse.ArgumentParser() with all the arguments setup.
    """
    parser = ArgumentParser()

    parser.add_argument('--type', '-t', type=str, required=True, help='What type of model you want to train (CNN, Grid classication)')
    parser.add_argument('--train', '-tm', action='store_true', help='This flag will train the model')
    parser.add_argument('-d', '--data', type=str, required=True, help='Path for the data')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose print debugging lines')
    parser.add_argument('-da', '--data_arch', type=str, default='eyeq', help='Data format type e.g. eyeq, mit')
    parser.add_argument('--test', '-vm', action='store_true', help='This flag will allow valdation of a model.')
    parser.add_argument('-e', '--epoch', type=int, default=10, help='Number of epochs to run model with')
    parser.add_argument('-bs', '--batch_size', type=int, default=64, help='Batch Size to run model with')
    parser.add_argument('--modelLoc', '-ml', type=str, help='The location where you model is stored.')
    parser.add_argument('--retrain', '-rt', action='store_true', help='Used to flag for retraining a model')
    parser.add_argument('--progressFile', '-p', type=str, help='The name of the progress file e.g. Model-001 (Will overwrite progress files with the same name)')
    parser.add_argument('--name', '-n', type=str, help='The name for the model, only use letters', default='eyeq')
    parser.add_argument('--screenSizeX', '-ssx', type=int, help='Size of screen to use for GCNN Model - X Resolution - (GCNN Only Argument)')
    parser.add_argument('--screenSizeY', '-ssy', type=int, help='Size of screen to use for GCCN Model - Y Resolution - (GCNN Only Argument)')
    parser.add_argument('--gridRowSize', '-grs', type=int, help='Size of grid for GCNN Model per row (GCCN Only Argument)')
    parser.add_argument('--gridColSize', '-gcs', type=int, help='Size of grid for GCNN Model per column(GCNN Only Argument)')
    if args_data:
        return parser.parse_args(args_data)
    else:
        return parser.parse_args()


def load_metadata_mit(file_loc, val_only=False):
    """
        Load the data file from the users input, this is formatted to mit layout.

        Args:
            file_loc (str): The string of the file location.
            val_only (bool): flag for returing only validation data.

        Return:
            train_data (List): List of the train values.
            validation_data (List): List of the validation values.
    """
    eye_data = np.load(file_loc)
    train_data = [eye_data["train_eye_left"],
                  eye_data['train_eye_right'],
                  eye_data['train_face'],
                  eye_data["train_face_mask"],
                  eye_data["train_y"]]
    validation_data = [eye_data["val_eye_left"],
                       eye_data['val_eye_right'],
                       eye_data['val_face'],
                       eye_data["val_face_mask"],
                       eye_data["val_y"]]
    if val_only:
        return validation_data
    return train_data, validation_data


def load_metadata_eyeq(file_loc, val_only=False, gcnn=False, x_res=None, y_res=None, row_size=None, col_size=None):
    """
        Load the data file from the users input, this is formatted to eyeq layout.

        Args:
            file_loc (str): The string of the file location.
            val_only (bool): flag for returing only validation data.
            gcnn: Boolean to check if gcnn is being used to train or test model

        Return:
            train_data (List): List of the train values.
            validation_data (List): List of the validation values.
    """
    eye_data = np.load(file_loc)
    if gcnn:
        y_label_train = []
        y_label_validation = []
        for x in range(len(eye_data["train_xPos"])):
            y_label_train.append((eye_data["train_xPos"][x], eye_data["train_yPos"][x]))
        for y in range(len(eye_data["val_xPos"])):
            y_label_validation.append((eye_data["val_xPos"][y], eye_data["val_yPos"][y]))

        if not x_res or not y_res or not row_size or not col_size:
            train_data = [eye_data["train_left_eye"],
                          eye_data['train_right_eye'],
                          eye_data['train_face'],
                          eye_data["train_face_mask"],
                          val_to_grid(y_label_train)]

            validation_data = [eye_data["val_left_eye"],
                               eye_data['val_right_eye'],
                               eye_data['val_face'],
                               eye_data["val_face_mask"],
                               val_to_grid(y_label_validation)]
        else:
            train_data = [eye_data["train_left_eye"],
                          eye_data['train_right_eye'],
                          eye_data['train_face'],
                          eye_data["train_face_mask"],
                          val_to_grid(y_label_train, x_res, y_res, row_size, col_size)]

            validation_data = [eye_data["val_left_eye"],
                               eye_data['val_right_eye'],
                               eye_data['val_face'],
                               eye_data["val_face_mask"],
                               val_to_grid(y_label_validation, x_res, y_res, row_size, col_size)]

    else:
        train_data = [eye_data["train_left_eye"],
                      eye_data['train_right_eye'],
                      eye_data['train_face'],
                      eye_data["train_face_mask"],
                      np.column_stack((eye_data["train_xPos"], eye_data["train_yPos"]))]
        validation_data = [eye_data["val_left_eye"],
                           eye_data['val_right_eye'],
                           eye_data['val_face'],
                           eye_data["val_face_mask"],
                           np.column_stack((eye_data["val_xPos"], eye_data["val_yPos"]))]
    if val_only:
        return validation_data
    else:
        return train_data, validation_data


def load_metadata_npza(file_loc, val_only=False):
    """
        Load the data file from the users input, this is formatted to npza layout.

        Args:
            file_loc (str): The string of the file location.
            val_only (bool): flag for returing only validation data.

        Return:
            train_data (List): List of the train values.
            validation_data (List): List of the validation values.
    """

    eye_data = np.load(file_loc)
    train_data = [eye_data["train_left_eye"],
                  eye_data['train_right_eye'],
                  eye_data['train_face'],
                  eye_data["train_face_mask"],
                  np.column_stack((eye_data["train_xPos"], eye_data["train_yPos"]))]

    validation_data = [eye_data["val_left_eye"],
                       eye_data['val_right_eye'],
                       eye_data['val_face'],
                       eye_data["val_face_mask"],
                       np.column_stack((eye_data["val_xPos"], eye_data["val_yPos"]))]

    if val_only:
        return validation_data

    return train_data, validation_data


def normalize(data):
    """
        The data must be normalize so that it does not break the net architecture

        Args:
             Data (numpy.array): Data to transform.

        Returns:
            numpy.array: Data reshaped to fit the network architecture.
    """
    pre_shape = data.shape
    data = np.reshape(data, (pre_shape[0], -1))
    data = data.astype('float32') / 255.0  # Used to scale
    data = data - np.mean(data, axis=0)  # normalizing the data
    return np.reshape(data, pre_shape)


def format_data(data):
    """
        Format the data so that it can be pased to the model and be used.

        Args:
            data (List): Input data to modified.

        Returns
            List: The input data formatted.
    """
    eye_l, eye_r, face, mask, prediction = data
    eye_l = normalize(eye_l)
    eye_r = normalize(eye_r)
    face = normalize(face)
    # The mask shape has to be reshape as the net can not take the (10, 25, 25):
    # Shape must be rank 2 but is rank 3 for 'MatMul_2' (op: 'MatMul') with input shapes: [?,25,25], [625,256]
    # So we need to reshape the data to a (?, 625) input.
    mask = np.reshape(mask, (mask.shape[0], -1)).astype('float32')
    prediction = prediction.astype('float32')
    return [eye_l, eye_r, face, mask, prediction]


def train(settings):
    '''
        Taking a value type and creating the correct model and then calling the train funcation of that model.

        Args:
            settings (ArgumentParser): Contains the users input.

        Raise:
            ValueError: If type is not supported by the models.
    '''
    if settings.data_arch.lower() == 'eyeq':
        if settings.type.lower() == 'gcnn':
            x_res = settings.screenSizeX
            y_res = settings.screenSizeY
            row_size = settings.gridRowSize
            col_size = settings.gridColSize
            train, validation = load_metadata_eyeq(settings.data, gcnn=True, x_res=x_res, y_res=y_res, row_size=row_size, col_size=col_size)
        else:
            train, validation = load_metadata_eyeq(settings.data)

    elif settings.data_arch.lower() == 'mit':
        train, validation = load_metadata_mit(settings.data)

    train = format_data(train)
    validation = format_data(validation)
    epochs = settings.epoch
    batch_size = settings.batch_size
    model_loc = settings.modelLoc

    if settings.type.lower() == 'cnn':
        eyeQ = Cnn_regression(settings.verbose, re_train=settings.retrain, progress_filename=settings.progressFile)
    elif settings.type.lower() == 'gcnn':
        if settings.gridRowSize:
            grid_size = settings.gridRowSize * settings.gridColSize
            eyeQ = Gccn_classification(grid_size=grid_size)
        else:
            eyeQ = Gccn_classification()
    else:
        types = 'CNN - Regression model, GCNN - Classification Model'
        err = 'Model input {} type was not found please try one of the following: {} '.format(settings.type, types)
        raise ValueError(err)

    if not settings.retrain:
        eyeQ.train(train, validation, epochs=epochs, batch_size=batch_size, name=settings.name.lower())
    else:
        eyeQ.train(train, validation, retrain_path=settings.modelLoc)
    if settings.verbose:
        print('Done training model')


def test(settings):
    """
        Taking settings and running a already train model against a set of data.

        Args:
            settings (ArgumentParser): Contains the users input.

        Raise:
            ValueError: If type is not supported by the models.

        Returns:
            (json) the prediction data.
    """
    if settings.data_arch.lower() == 'eyeq':
        if settings.type.lower() == 'gcnn':
            x_res = settings.screenSizeX
            y_res = settings.screenSizeY
            row_size = settings.gridRowSize
            col_size = settings.gridColSize
            train, validation = load_metadata_eyeq(settings.data, gcnn=True, x_res=x_res, y_res=y_res, row_size=row_size, col_size=col_size)
        else:
            train, validation = load_metadata_eyeq(settings.data)

    elif settings.data_arch.lower() == 'mit':
        validation = load_metadata_mit(settings.data, val_only=True)

    validation = format_data(validation)

    if settings.type.lower() == 'cnn':
        eyeQ = Cnn_regression(settings.verbose, False)
    elif settings.type.lower() == 'gcnn':
        if settings.gridRowSize:
            grid_size = settings.gridRowSize * gridColSize
            eyeQ = Gccn_classification(grid_size=grid_size)
        else:
            eyeQ = Gccn_classification()
    else:
        types = 'CNN - Regression model, GCNN - Classification Model'
        err = 'Model input {} type was not found please try one of the following: {} '.format(settings.type, types)
        raise ValueError(err)

    return eyeQ.testing(settings.modelLoc, validation, settings.batch_size, settings.name)


def get_filepaths():
    """
        Returns the location of all the file paths that the module uses.

        Return:
            (dict) {'File': 'path_to_file'}
    """
    realpath = os.path.dirname(os.path.realpath(__file__))
    return {
        'models': '{}{}eye_q'.format(realpath, os.sep),
        'progress_files': '{}{}progress'.format(realpath, os.sep),
        'testing_files': '{}{}test_data'.format(realpath, os.sep),
    }


if __name__ == "__main__":
    main()
