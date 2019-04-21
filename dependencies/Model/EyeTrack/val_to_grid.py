import numpy as np
import math


def val_to_grid(labels, x_res=1366, y_res=1024, row_size=6, col_size=4):
    '''
     A script to convert label values to a position in a grid for
     classification model

     Args:
        labels: Labels from MIT or EyeQ style datasets
        x_res: Horizontal Resolution of target device - Default: 1200
        y_res: Vertical Resolution of target device - Default: 800
        row_size: Size of each row in the grid - Default: 6
        col_size: Size of each column in the grid - Default: 4

     Returns:
        A numpy array of labels for classification model
    '''
    # Split Resolution evenly
    row_size -= 1
    col_size -= 1
    x_res_per_grid_pos = x_res / row_size
    y_res_per_grid_pos = y_res / col_size

    x_classification_labels = []
    y_classification_labels = []

    # Iterate over every position in lables
    for val in labels:
        x_label = int(val[0] / x_res_per_grid_pos)
        y_label = int(val[1] / y_res_per_grid_pos)

        # Bit of a hack around to deal with orientation change
        if y_label <= col_size:
            x_classification_labels.append(x_label)
            y_classification_labels.append(y_label)
        else:
            x_classification_labels.append(y_label)
            y_classification_labels.append(x_label)

    classification_labels = []

    # Adding labels together to get grid position
    for i in range(len(x_classification_labels)):
        label = (x_classification_labels[i] + (y_classification_labels[i] * row_size))
        classification_labels.append(label)

    classification_labels = np.asarray(classification_labels)

    return classification_labels
