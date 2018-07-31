"""
By Shah-Rukh
"""

import numpy as np
import helpers

DIM_1 = 16
DIM_2 = 16
DIM_3 = 55
HALF_ETA = 0.2
HALF_PHI = 0.2
MAX_ELEMENTS=6

BIN_WIDTH_ETA = 2 * HALF_ETA / DIM_1
BIN_WIDTH_PHI = 2 * HALF_PHI / DIM_2

def _find_indices(histo, eta_bins, phi_bins, layers):
    n = np.size(eta_bins)
    indices = np.zeros(n, dtype=np.int32)
    for i in range(n):
        if eta_bins[i] >= 0 and eta_bins[i] < DIM_1 and phi_bins[i] >= 0 and phi_bins[i] < DIM_2 and layers[i] >= 0 and \
                layers[i] < DIM_3:
            index = histo[eta_bins[i], phi_bins[i], layers[i]]
            histo[eta_bins[i], phi_bins[i], layers[i]] += 1
        else:
            index = -1
        indices[i] = index
    indices[indices >= 6] = -1
    return indices


def process(rechit_eta, rechit_phi, rechit_layer, rechit_energy, seed_eta, seed_phi):
    eta_low_edge = seed_eta - HALF_ETA
    eta_bins = np.floor((rechit_eta - eta_low_edge) / BIN_WIDTH_ETA).astype(np.int32)

    phi_low_edge = helpers.delta_angle(seed_phi, HALF_PHI)
    phi_bins = np.floor(helpers.delta_angle(rechit_phi, phi_low_edge) / BIN_WIDTH_PHI).astype(np.int32)

    layers = np.minimum(np.floor(rechit_layer) - 1, 54).astype(np.int32)

    histogram = np.zeros((DIM_1,DIM_2,DIM_3))
    indices = _find_indices(histogram, eta_bins, phi_bins, layers)
    indices_valid = np.where(indices!=-1)
    store_energy = rechit_energy[indices_valid]
    store_eta_bins = eta_bins[indices_valid]
    store_phi_bins = phi_bins[indices_valid]
    store_layers = layers[indices_valid]

    data_x = np.zeros((DIM_1, DIM_2, DIM_3, 6))
    data_x[store_eta_bins, store_phi_bins, store_layers, indices[indices_valid]] = store_energy

    return data_x
