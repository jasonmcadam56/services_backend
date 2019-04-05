import numpy as np


def val_to_grid(labels):
    '''
     A script to convert label values to a position in a grid for
     classification model

     Args:
        labels: Labels from MIT or EyeQ style datasets

     Returns:
        A numpy column stack to be used as labels for classification model
    '''
    # Resolution of target device, could take this in as a variable later on
    X_RES = 1200
    Y_RES = 800

    # Split Resolution evenly
    x_res_per_grid_pos = X_RES / 6
    y_res_per_grid_pos = Y_RES / 4

    x_grid_array = np.array([x for x in range(0, X_RES, int(x_res_per_grid_pos))])
    y_grid_array = np.array([y for y in range(0, Y_RES, int(y_res_per_grid_pos))])

    # Adding final sector value to array
    x_grid_array = np.append(x_grid_array, X_RES)
    y_grid_array = np.append(y_grid_array, Y_RES)

    x_classification_labels = []
    y_classification_labels = []

    # Iterate over every position in lables
    for val in labels:
        # Need to handle orientation changes
        if val[1] > Y_RES:
            # Iterate over values in x grid array
            for x in range(len(x_grid_array)):
                if val[1] >= x_grid_array[x] and val[1] < x_grid_array[x + 1]:
                    print("Y Inverted")
                    print("Between {} and {}".format(x_grid_array[x], x_grid_array[x + 1]))
                    print("Pos: {}".format(val[1]))
                    print("Break at {}\n".format(x))
                    x_classification_labels.append(x)
                    break

            # Iterate over values in y grid array
            for y in range(len(y_grid_array)):
                if val[0] >= y_grid_array[y] and val[0] < y_grid_array[y + 1]:
                    print("X Inverted")
                    print("Between {} and {}".format(y_grid_array[y], y_grid_array[y + 1]))
                    print("Pos: {}".format(val[0]))
                    print("Break at {}\n".format(y))
                    y_classification_labels.append(y)
                    break

        else:
            # Iterate over values in x grid array
            for x in range(len(x_grid_array)):
                if val[0] >= x_grid_array[x] and val[0] < x_grid_array[x + 1]:
                    print("X Normal")
                    print("Between {} and {}".format(x_grid_array[x], x_grid_array[x + 1]))
                    print("Pos: {}".format(val[0]))
                    print("Break at {}\n".format(x))
                    x_classification_labels.append(x)
                    break
            # Iterate over values in y grid array
            for y in range(len(y_grid_array)):
                if val[1] >= y_grid_array[y] and val[1] < y_grid_array[y + 1]:
                    print("Y Normal")
                    print("Between {} and {}".format(y_grid_array[y], y_grid_array[y + 1]))
                    print("Pos: {}".format(val[1]))
                    print("Break at {}\n".format(y))
                    y_classification_labels.append(y)
                    break

    classification_labels = []
    print(len(x_classification_labels))
    print(len(y_classification_labels))
    # Adding labels together to get grid position

    classification_labels = np.column_stack((x_classification_labels, y_classification_labels))

    return classification_labels
