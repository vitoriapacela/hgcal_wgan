import root_numpy
from binner_energy import process
import glob
import h5py
import numpy as np
import os

FOLDER_NAME = "gamma/"

def convert(f1):
    '''
    Converts data to images, only for the electrons in the dataset.
    :param f1: root file name
    :type f1: str
    :return: images, true_energy, and isPileup arrays
    :rtype: three numpy arrays
    '''
    
    branches = ['isElectron', 'isMuon', 'isPionCharged', 'isPionNeutral', 'isK0Long', 'isK0Short', 'isGamma', 'true_energy', 'true_x', 'true_y', 'true_r', 'rechit_energy', 'rechit_x', 'rechit_y', 'rechit_z', 'rechit_layer', 'rechit_varea', 'rechit_vz', 'rechit_vxy', 'rechit_detid']
    A = root_numpy.root2array(f1, treename='B4;1', branches=branches)
    
    # get indexes
    isGamma = np.where(A['isGamma']==1)

    # select only electrons
    A = A[isGamma]

    #print(A.shape)
    num_entries = len(A)
    print("Number of entries is ", num_entries)
    
    # N rechits per event
    rechit_energy = A['rechit_energy'] # N energy
    rechit_x = A['rechit_x']
    rechit_y = A['rechit_y']
    rechit_layer = A['rechit_layer']

    seed_x = A['true_x'] # scalar x of the seed
    seed_y = A['true_y'] # scalar y of the seed

    ## other data:
    true_energy = A['true_energy']

    pics = []
    for i in range(num_entries):
        pic4d = process(rechit_x[i], rechit_y[i], rechit_layer[i], rechit_energy[i], seed_x[i], seed_y[i])
        reduced = np.sum(np.abs(pic4d), axis=3)
        pics.append(reduced)

    return np.asarray(pics), true_energy


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
    #h5f_no_pu.flush()
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
