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


def main(from_app=[]):
    """
        Process the user input and set up the correct model and pass in the data or use a pre-trained model for processing data.

        kargs:
            type    : Model type (At moment only CNN supported)
            train   : If the program should be in train mode.
            data:   : Location of the image data.
            verbose : Allow debug statments to be printed to the console.
    """
    settings = process_args(from_app)

    if settings.verbose:
        print('Verbose mode is on.')

    data_archs = ['eyeq', 'mit']

    if settings.data_arch not in data_archs:
        raise ValueError('Data architecture must be either {}'.format(data_archs))

    if settings.train:
        train(settings)
    elif settings.test:
        test(settings)
    else:
        print("training off")


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
    parser.add_argument('--modelLoc', '-ml', type=str, help='The location where you model is stored.')

    if args_data:
        return parser.parse_args(args_data)
    else:
        return parser.parse_args()


def load_metadata_mit(file_loc, val_only=False):

    """
        Load the data file from the users input, this is formatted to mit layout.

        Args:
            file_loc (str): The string of the file location.

        Return:
            train_data (List): List of the train values.
            valation_data (List): List of the valation values.
    """
    eye_data = np.load(file_loc)
    train_data = [eye_data["train_eye_left"],
                  eye_data['train_eye_right'],
                  eye_data['train_face'],
                  eye_data["train_face_mask"],
                  eye_data["train_y"]]
    valation_data = [eye_data["val_eye_left"],
                     eye_data['val_eye_right'],
                     eye_data['val_face'],
                     eye_data["val_face_mask"],
                     eye_data["val_y"]]
    if val_only:
        return valation_data
    return train_data, valation_data


def load_metadata_eyeq(file_loc, val_only=False):

    """
        Load the data file from the users input, this is formatted to eyeq layout.

        Args:
            file_loc (str): The string of the file location.

        Return:
            train_data (List): List of the train values.
            valation_data (List): List of the valation values.
    """
    eye_data = np.load(file_loc)
    train_data = [eye_data["train_left_eye"],
                  eye_data['train_right_eye'],
                  eye_data['train_face'],
                  eye_data["train_face_mask"],
                  np.column_stack((eye_data["train_xPos"], eye_data["train_yPos"]))]

    valation_data = [eye_data["val_left_eye"],
                     eye_data['val_right_eye'],
                     eye_data['val_face'],
                     eye_data["val_face_mask"],
                     np.column_stack((eye_data["val_xPos"], eye_data["val_yPos"]))]

    if val_only:
        return valation_data

    return train_data, valation_data


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
        train, valation = load_metadata_eyeq(settings.data)
    elif settings.data_arch.lower() == 'mit':
        train, valation = load_metadata_mit(settings.data)

    train = format_data(train)
    valation = format_data(valation)

    if settings.type.lower() == 'cnn':
        eyeQ = Cnn_regression(settings.verbose)
    else:
        types = 'CNN - Regression model'
        err = 'Model input {} type was not found please try one of the following: {} '.format(settings.type, types)
        raise ValueError(err)

    eyeQ.train(train, valation)
    if settings.verbose:
        print('Done training model')


def test(settings):
    """
        Taking settings and running a already train model against a set of data.

        Args:
            settings (ArgumentParser): Contains the users input.

        Raise:
            ValueError: If type is not supported by the models.
    """
    if settings.data_arch.lower() == 'eyeq':
        valation = load_metadata_eyeq(settings.data, val_only=True)
    elif settings.data_arch.lower() == 'mit':
        valation = load_metadata_mit(settings.data, val_only=True)

    valation = format_data(valation)

    if settings.type.lower() == 'cnn':
        eyeQ = Cnn_regression(settings.verbose, False)
    else:
        types = 'CNN - Regression model'
        err = 'Model input {} type was not found please try one of the following: {} '.format(settings.type, types)
        raise ValueError(err)

    err, mape = eyeQ.testing(settings.modelLoc, valation)
    print('Overall Error rate {}'.format(err))
    print('Overall Mean absolute percentage error {}'.format(mape))


if __name__ == "__main__":
    main()
