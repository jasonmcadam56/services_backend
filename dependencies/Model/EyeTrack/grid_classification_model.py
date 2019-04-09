'''
    A classificaiton eye tracking model built using Keras that will
    map gaze coordinates to screen location. Based of MIT Gaze Capture
    Model but instead of getting a screen position, instead we get a larger
    grid based position of where the user is looking
'''
import os
import json
import math

import numpy as np

from keras import backend
from keras.layers import Layer, Input, Conv2D, MaxPool2D, Flatten, Dense, concatenate
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam
from keras.callbacks import Callback, ModelCheckpoint, BaseLogger, ProgbarLogger, TensorBoard, ReduceLROnPlateau, EarlyStopping
from keras.utils import to_categorical


class JSONRecorder(Callback):
    '''
        Custom callback to record model data to JSON File
        while training

        Args:
            name: Name of JSON file to save to
            model_loc: Location model is save to
    '''

    def __init__(self, name, model_loc):
        super(JSONRecorder, self).__init__()
        self.name = "{}.json".format(name)
        self.model_loc = model_loc

    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']
        self.best_loss = math.inf
        self.stats = ''

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('loss')
        if current_loss < self.best_loss:
            self.best_loss = current_loss
        progress_dict = {"starting_epoch": 1,
                         "current_epoch": epoch + 1,
                         'final_epoch': self.epochs,
                         'best_loss': self.best_loss,
                         'current_loss': current_loss}

        self.stats = logs

        with open(self.name, 'w') as prog_file:
            json.dump(progress_dict, prog_file)

    def on_train_end(self, logs=None):
        final_dict = {"status": "complete",
                      "model_loc": os.path.abspath(self.model_loc),
                      "stats": str(self.stats),
                      "model_name": os.path.splitext(self.model_loc)[0]}

        with open(self.name, 'w') as final_file:
            json.dump(final_dict, final_file)


def create_eye_model(image_columns, image_rows, image_channel):
    '''
        Create an Convolutional model for eye data
        Will be called twice due to needing 2 seperate models for each eye

        Args:
            image_columns - Data representing image data column
            image_rows - Data representing image data row
            image_channel - Data representing image data for colour channels
        Returns:
            A connected Keras Model for a single eye
    '''
    # Take in an eye image
    eye_image = Input(shape=(image_columns, image_rows, image_channel))

    eye = Conv2D(filters=96, kernel_size=(11, 11), activation='relu', data_format='channels_last')(eye_image)
    eye = MaxPool2D(pool_size=(2, 2))(eye)
    eye = Conv2D(filters=256, kernel_size=(5, 5), activation='relu', data_format='channels_last')(eye)
    eye = MaxPool2D(pool_size=(2, 2))(eye)
    eye = Conv2D(filters=384, kernel_size=(3, 3), activation='relu', data_format='channels_last')(eye)
    eye = MaxPool2D(pool_size=(2, 2))(eye)

    output = Conv2D(filters=64, kernel_size=(1, 1), activation='relu', data_format='channels_last')(eye)

    model = Model(
        inputs=eye_image,
        outputs=output)

    return model


def create_face_model(image_columns, image_rows, image_channel):
    '''
        Create an Convolutional model for face data

        Args:
            image_columns - Data representing image data column
            image_rows - Data representing image data row
            image_channel - Data representing image data for colour channels
        Returns:
            A connected Keras Model for a face data
    '''
    face_image = Input(shape=(image_columns, image_rows, image_channel))

    face = Conv2D(filters=96, kernel_size=(11, 11), activation='relu', data_format='channels_last')(face_image)
    face = MaxPool2D(pool_size=(2, 2))(face)
    face = Conv2D(filters=256, kernel_size=(5, 5), activation='relu', data_format='channels_last')(face)
    face = MaxPool2D(pool_size=(2, 2))(face)
    face = Conv2D(filters=384, kernel_size=(3, 3), activation='relu', data_format='channels_last')(face)
    face = MaxPool2D(pool_size=(2, 2))(face)

    output = Conv2D(filters=64, kernel_size=(1, 1), activation='relu', data_format='channels_last')(face)

    model = Model(
        inputs=face_image,
        outputs=output)

    return model


def create_eye_tracking_model(image_columns, image_rows, image_channel, grid_size):
    '''
        Connect models and finnally output to final layer for classification

        Args:
            image_columns - Data representing image data column
            image_rows - Data representing image data row
            image_channel - Data representing image data for colour channels
            grid_size = Size of grid to be used for classification

        Returns:
            A fully connected Keras Model

    '''
    eye_network = create_eye_model(image_columns, image_rows, image_channel)
    face_network_partial = create_face_model(image_columns, image_rows, image_channel)

    right_eye_input = Input(shape=(image_columns, image_rows, image_channel))
    right_eye_network = eye_network(right_eye_input)

    left_eye_input = Input(shape=(image_columns, image_rows, image_channel))
    left_eye_network = eye_network(left_eye_input)

    face_input = Input(shape=(image_columns, image_rows, image_channel))
    face_network = face_network_partial(face_input)

    face_grid = Input(shape=(25 * 25, ))

    eyes = concatenate([left_eye_network, right_eye_network])
    eyes = Flatten()(eyes)
    fc_e1 = Dense(units=128, activation='relu')(eyes)

    face = Flatten()(face_network)
    fc_f1 = Dense(units=128, activation='relu')(face)
    fc_f2 = Dense(units=64, activation='relu')(fc_f1)

    fc_fg1 = Dense(units=256, activation='relu')(face_grid)
    fc_fg2 = Dense(units=128, activation='relu')(fc_fg1)

    final_dense_layer = concatenate([fc_e1, fc_f2, fc_fg2])

    fc1 = Dense(units=128, activation='relu')(final_dense_layer)
    fc2 = Dense(units=grid_size, activation='softmax')(fc1)

    final_model = Model(
        inputs=[right_eye_input, left_eye_input, face_input, face_grid],
        output=fc2)

    return final_model


def train(training, validation, epochs, batch_size, grid_size=24, name='', model_loc=''):
    '''
        Train the model based on trainging and validation data
        When training is finished, a model will be saved to a
        .h5 file

        Args:
            training: A numpy array of training data
            validation: A numpy array of validation data
            epochs: Number of epochs to train up to
            batch_size: Number of batches to trian on per epoch
            grid_size: Size of grid to train model against
            model_loc: If a model has been previously trained, pass in the path
                       and continue training on it
    '''
    print("Now training using Grid Classification")
    image_columns = 64
    image_rows = 64
    image_channel = 3
    train_y = training[4]
    val_y = validation[4]
    train_class_labels = to_categorical(train_y, num_classes=grid_size)
    val_class_labels = to_categorical(val_y, num_classes=grid_size)
    training = training[:4]

    if model_loc is "":
        gcnn = create_eye_tracking_model(image_columns, image_rows, image_channel, grid_size)
        if not os.path.exists('grid/'):
            os.makedirs('grid/')
        adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        gcnn.compile(optimizer=adam,
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        model_loc = 'grid/gcnn.h5'

    else:
        model = load_model(model_loc)
        config = model.get_config()
        gcnn = Model.from_config(config)

    gcnn.summary()

    gcnn.fit(x=training,
             y=train_class_labels,
             batch_size=batch_size,
             epochs=epochs,
             verbose=1,
             shuffle=True,
             validation_data=(validation[:4], val_class_labels),
             callbacks=[ModelCheckpoint(model_loc, monitor="loss", verbose=1, save_best_only=True),
                        ProgbarLogger(),
                        TensorBoard(log_dir='grid/logs'),
                        ReduceLROnPlateau(),
                        EarlyStopping(monitor="loss", patience=10),
                        JSONRecorder(name, model_loc)])

    gcnn.save(model_loc)

    eval_loss, eval_acc = gcnn.evaluate(x=training,
                                        y=train_class_labels)

    print("GCNN Loss:\t{}\nGCNN Accuracy:\t{}".format(eval_loss, eval_acc))


def testing(model_loc, data):
    '''
        Take a created model file and test it against unseen data to get
        accuracy

        Args:
            model_loc: Path to previously created model
            data: Path to data to test on

        Raise:
            FileNotFoundError: Raise if path to model passed in is wrong
    '''
    if not os.path.exists(model_loc):
        raise FileNotFoundError("Model file {} does not exist".format(model_loc))

    gcnn = load_model(model_loc)
    y = data[4]
    x_error = []
    y_error = []
    gcnn_prob = gcnn.predict(x=data[:4], verbose=1, batch_size=1)
    gcnn_classes = gcnn_prob.argmax(axis=-1)
    for i, prob in enumerate(gcnn_prob):
        print("\nGrid Pos {} with probability {}%".format(prob.argmax(axis=-1), (prob[prob.argmax(axis=-1)] * 100)))
        print("Label: {}".format(y[i]))
    x_error = 1
    y_error = 1

    mase = (x_error, y_error)

    return mase, 0.0
