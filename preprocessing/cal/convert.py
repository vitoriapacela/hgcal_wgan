import root_numpy
import glob
import h5py
import numpy as np
import os

# folder to save the output files
FOLDER_NAME = "gamma/"

def convert(f1):
    '''
    Converts data to images, only for a certain particle (e.g. electrons, photons) in the dataset.
    :param f1: root file name
    :type f1: str
    :return: images, true_energy, and isPileup arrays
    :rtype: three numpy arrays
    '''
    
    branches = ['isElectron', 'isMuon', 'isPionCharged', 'isPionNeutral', 'isK0Long', 'isK0Short', 'isGamma', 'true_energy', 'true_x', 'true_y', 'true_r', 'rechit_energy', 'rechit_x', 'rechit_y', 'rechit_z', 'rechit_layer', 'rechit_varea', 'rechit_vz', 'rechit_vxy', 'rechit_detid']
    A = root_numpy.root2array(f1, treename='B4;1', branches=branches)
    
    # get indexes
    isGamma = np.where(A['isGamma'] == 1)

    # select only electrons
    A = A[isGamma]

    num_entries = len(A)
    print("Number of entries is ", num_entries)
    
    # N rechits per event
    rechit_energy = A['rechit_energy'] # N energy
    rechit_x = A['rechit_x']
    rechit_y = A['rechit_y']
    rechit_layer = A['rechit_layer']

    ## other data:
    true_energy = A['true_energy']

    pics = []
    # loop through all the events in a file
    for i in range(num_entries):
        image = process(rechit_energy[i], rechit_x[i], rechit_y[i], rechit_layer[i])
        pics.append(image)

    return np.asarray(pics), true_energy

def process(rechit_energy_i, rechit_x_i, rechit_y_i, rechit_layer_i):
    '''
    Maps the energy reconstructed hits to a 3D array.
    :param rechit_energy_i: rechit energies of one event
    :type rechit_energy_i: numpy ndarray
    :param rechit_x_i: rechit xs of one event
    :type rechit_x_i: numpy ndarray
    :param rechit_y_i: rechit ys of one event
    :param rechit_layer_i: rechit layers of one event
    :type rechit_layer_i: numpy ndarray
    :return: 3D array of shape (10, 10, 30)
    :rtype: numpy ndarray
    '''
    #xs = np.unique(rechit_x_i)
    #ys = np.unique(rechit_y_i)
    #zs = np.unique(rechit_layer_i)
    
    #dim_x = len(xs)
    #dim_y = len(ys)
    #dim_z = len(zs)
   
    #xi = range(dim_x)
    #yi = range(dim_y)
    #zi = range(dim_z)

    #dict_x = dict(zip(xs, xi))  
    # since xs == ys, no need for dict_y

    ## create custom dict to map the rechit_x/y energy to its respective index
    dict_x = {-135:0, -131.25:0, -105:1, -93.75:1, -75:2, -56.25:3, -45:3, -18.75:4, -15:4, 0:5, 15:5, 18.75:5, 45:6, 56.25:6, 75:7, 93.75:8, 105:8, 131.25:9, 135:9}

    #D = np.zeros((dim_x, dim_y, dim_z))
    D = np.zeros((10, 10, 30))

    ## Map each energy rechit in an event to its relative position in a 3D array
    ## and convert the rechit energy from MeV to GeV
    for ind, energy in enumerate(rechit_energy_i):
        D[dict_x[rechit_x_i[ind]], dict_x[rechit_y_i[ind]], int(rechit_layer_i[ind])] = energy / 1000

    return D


def toH5(f1, pics, trues):
    '''
    Creates pileup and no-pileup HDF5 files containing image and true_energy arrays from a root file.
    :param f1: root file name
    :type f1: str
    :param pics: images created from rechit parameters
    :type pics: numpy array
    :param trues: ground truth energy values
    :type param: numpy array
    '''
    file_name = FOLDER_NAME + f1.split('/')[-1].split('.')[0] + '.h5'
    print(file_name)

    h5f_no_pu = h5py.File(file_name, 'w')
    
    h5f_no_pu.create_dataset('image', data=pics)
    h5f_no_pu.create_dataset('true_energy', data=trues)

    h5f_no_pu.close()


def main():
    d = '/eos/cms/store/cmst3/group/dehep/miniCalo/pi0gamma_homog/'

    root_files = glob.glob(d + '*.root')

    #root_files.remove('/eos/cms/store/cmst3/group/dehep/miniCalo/pi0gamma_homog/2_101_out.root')

    for f in root_files:
        pics, trues = convert(f)
        toH5(f, pics, trues)
        
    print("Done!")


if __name__ == "__main__":
    if not os.path.exists(os.getcwd() + "/" + FOLDER_NAME):
        os.makedirs(os.getcwd() + "/"+ FOLDER_NAME)
    main()
