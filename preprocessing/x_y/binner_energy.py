"""
Adapted from Shah Rukh's code.
Creates a 3-D array of zeroes and maps the rechits to their respective positions.
"""

import numpy as np

DIM_1 = 16
DIM_2 = 16
DIM_3 = 52
HALF_X = 270
# HALF_X = np.max(rechit_x[0]) - np.min(rechit_x[0])
HALF_Y = 270
MAX_ELEMENTS=6

BIN_WIDTH_X = 2 * HALF_X / DIM_1
BIN_WIDTH_Y = 2 * HALF_Y / DIM_2

def _find_indices(histo, x_bins, y_bins, layers):
    n = np.size(x_bins)
    indices = np.zeros(n, dtype=np.int32)
    for i in range(n):
        if x_bins[i] >= 0 and x_bins[i] < DIM_1 and y_bins[i] >= 0 and y_bins[i] < DIM_2 and layers[i] >= 0 and \
                layers[i] < DIM_3:
            index = histo[x_bins[i], y_bins[i], layers[i]]
            histo[x_bins[i], y_bins[i], layers[i]] += 1
        else:
            index = -1
        indices[i] = index
    indices[indices >= 6] = -1
    return indices


def process(rechit_x, rechit_y, rechit_layer, rechit_energy, seed_x, seed_y):
    x_low_edge = seed_x - HALF_X
    x_bins = np.floor((rechit_x - x_low_edge) / BIN_WIDTH_X).astype(np.int32)

    y_low_edge = seed_y - HALF_Y
    y_bins = np.floor((rechit_y - y_low_edge) / BIN_WIDTH_Y).astype(np.int32)

    layers = np.minimum(np.floor(rechit_layer) - 1, 51).astype(np.int32)

    histogram = np.zeros((DIM_1, DIM_2, DIM_3))
    indices = _find_indices(histogram, x_bins, y_bins, layers)
    indices_valid = np.where(indices!=-1)
    store_energy = rechit_energy[indices_valid]
    store_x_bins = x_bins[indices_valid]
    store_y_bins = y_bins[indices_valid]
    store_layers = layers[indices_valid]

    data_x = np.zeros((DIM_1, DIM_2, DIM_3, 6))
    data_x[store_x_bins, store_y_bins, store_layers, indices[indices_valid]] = store_energy

    return data_x
