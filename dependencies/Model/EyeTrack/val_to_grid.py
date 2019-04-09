import numpy as np


def val_to_grid(labels, x_res=1200, y_res=800, row_size=6, col_size=4):
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
    x_res_per_grid_pos = x_res / row_size
    y_res_per_grid_pos = y_res / col_size

    x_grid_array = np.array([x for x in range(0, x_res, int(x_res_per_grid_pos))])
    y_grid_array = np.array([y for y in range(0, y_res, int(y_res_per_grid_pos))])

    # Adding final sector value to array
    x_grid_array = np.append(x_grid_array, x_res)
    y_grid_array = np.append(y_grid_array, y_res)

    x_classification_labels = []
    y_classification_labels = []

    # Iterate over every position in lables
    for val in labels:
        # Need to handle orientation changes
        if val[1] > y_res:
            # Iterate over values in x grid array
            for x in range(len(x_grid_array)):
                if val[1] >= x_grid_array[x] and val[1] < x_grid_array[x + 1]:
                    x_classification_labels.append(x)
                    break

            # Iterate over values in y grid array
            for y in range(len(y_grid_array)):
                if val[0] >= y_grid_array[y] and val[0] < y_grid_array[y + 1]:
                    y_classification_labels.append(y)
                    break

        else:
            # Iterate over values in x grid array
            for x in range(len(x_grid_array)):
                if val[0] >= x_grid_array[x] and val[0] < x_grid_array[x + 1]:
                    x_classification_labels.append(x)
                    break
            # Iterate over values in y grid array
            for y in range(len(y_grid_array)):
                if val[1] >= y_grid_array[y] and val[1] < y_grid_array[y + 1]:
                    y_classification_labels.append(y)
                    break

    classification_labels = []

    # Adding labels together to get grid position
    for i in range(len(x_classification_labels)):
        label = x_classification_labels[i] + (y_classification_labels[i] * col_size)
        classification_labels.append(label)

    classification_labels = np.asarray(classification_labels)

    return classification_labels
