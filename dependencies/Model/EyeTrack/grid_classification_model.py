'''
    A classificaiton eye tracking model built using Keras that will
    map gaze coordinates to screen location. Based of MIT Gaze Capture
    Model with one key difference, after the last fully connected layer
    we map the result to a grid representing screen locations

    TODO: - Training method
          - Test method
          - Add the classification after the last fully connected layer
'''
import os

import numpy as np
import tensorflow as tf

from keras import backend
from keras.layers import Layer, Input, Conv2D, MaxPool2D, Flatten, Dense, concatenate
from keras.models import Model, Sequential, load_model
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, BaseLogger, ProgbarLogger, TensorBoard, ReduceLROnPlateau, EarlyStopping


def create_eye_model(image_columns, image_rows, image_channel):
    '''
        Placeholder docstring
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
        Placeholder docstring
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


def create_eye_tracking_model(image_columns, image_rows, image_channel):
    '''
        Placeholder docstring
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

    # TODO - change model to sequential model

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
    fc2 = Dense(units=24, activation='softmax')(fc1)

    final_model = Model(
        inputs=[right_eye_input, left_eye_input, face_input, face_grid],
        outputs=[fc2])

    return final_model


def train(training, validation, epochs, batch_size, model_loc, path='', name=''):
    '''
        Placeholder docstring
    '''
    print("Now training using Grid Classification")
    image_columns = 64
    image_rows = 64
    image_channel = 3
    y = training[4]
    print(np.shape(y))
    training = training[:4]

    if model_loc is "":
        gcnn = create_eye_tracking_model(image_columns, image_rows, image_channel)
        if not os.path.exists('grid/'):
            os.makedirs('grid/')
        sgd = SGD(lr=1e-4, momentum=0.5, decay=0.1, nesterov=True)
        gcnn.compile(optimizer=sgd,
                     loss='mean_squared_error',
                     metrics=['accuracy'])
        model_loc = 'grid/gcnn.h5'

    else:
        gcnn = load_model(model_loc)

    gcnn.summary()

    gcnn.fit(x=training,
             y=y,
             batch_size=batch_size,
             epochs=epochs,
             verbose=1,
             shuffle=True,
             validation_data=(validation[:4], validation[4]),
             callbacks=[ModelCheckpoint(model_loc, monitor="loss", verbose=1, save_best_only=True),
                        ProgbarLogger(),
                        TensorBoard(log_dir='grid/logs'),
                        ReduceLROnPlateau(),
                        EarlyStopping(monitor="loss", patience=10)])

    gcnn.save(model_loc)

    eval_loss, eval_acc = gcnn.evaluate(x=training,
                                        y=y)

    print("GCNN Loss:\t{}\nGCNN Accuracy:\t{}".format(eval_loss, eval_acc))


def testing(model_loc, data):
    gcnn = load_model(model_loc)
    y = data[4]
    x_error = []
    y_error = []
    gcnn_pred = gcnn.predict(x=data[:4], verbose=1, batch_size=1)
    for i, pred in enumerate(gcnn_pred):
        print("\nPredict: {}".format(pred))
        print("Label: {}".format(y[i]))

        # x_error.append(abs(pred[0] - y[i][0]))
        # y_error.append(abs(pred[1] - y[i][1]))

    # x_mase=np.mean(x_error)
    # y_mase=np.mean(y_error)
    x_error = 1
    y_error = 1

    mase = (x_error, y_error)

    return mase, 0.0
