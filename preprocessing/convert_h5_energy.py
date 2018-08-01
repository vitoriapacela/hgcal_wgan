import root_numpy
from binner_energy import process
import glob
import h5py
import numpy as np
import os


def convert(f1):
    '''
    Converts data to images, only for the electrons in the dataset.
    :param f1: root file name
    :type f1: str
    :return: images, true_energy, and isPileup arrays
    :rtype: three numpy arrays
    '''
    
    branches = ['isGamma', 'isElectron', 'isMuon', 'isPionCharged', 'true_energy',
            'rechit_energy', 'rechit_phi', 'rechit_eta','rechit_layer', 'seed_eta', 'seed_phi', 'npu', 'true_ncloseparticles'
           ]

    A = root_numpy.root2array(f1, treename='deepntuplizer/tree', branches=branches)

    #isElectron = np.where( (A['isElectron']) and (A['true_ncloseparticles']==0) )
    
    isElectro = np.where(A['isElectron'])

    A = A[isElectro]

    ind = np.where(A['true_ncloseparticles']==0)

    # N rechits per event
    rechit_energy = A['rechit_energy'][ind]  # N energy
    rechit_eta = A['rechit_eta'][ind]
    rechit_phi = A['rechit_phi'][ind]
    rechit_layer = A['rechit_layer'][ind]

    seed_eta = A['seed_eta'][ind]  # scalar eta of the seed
    seed_phi = A['seed_phi'][ind]  # scalar phi of the seed

    ## other data:
    true_energy = A['true_energy'][ind]
    pu = A['npu'][ind]
    
    
    num_entries = len(ind[0])
    print("Number of entries is ", num_entries)

    pics = []
    for i in range(num_entries):
        pic4d = process(rechit_eta[i], rechit_phi[i], rechit_layer[i], rechit_energy[i], seed_eta[i], seed_phi[i])
        reduced = np.sum(np.abs(pic4d), axis=3)
        pics.append(reduced)

    return np.asarray(pics), true_energy, pu


def toH5(f1, pics, trues, pus):
    '''
    Creates pileup and no-pileup HDF5 files containing image and true_energy arrays from a root file.
    :param f1: root file name
    :type f1: str
    :param pics: images created from rechit parameters
    :type pics: numpy array
    :param trues: ground truth energy values
    :type param: numpy array
    :param pus: whether the event contains pileup (0 if no, 200 if yes)
    :type pus: numpy array
    '''
    
    file_name_pu = 'PU/' + f1.split('/')[11].split('.')[0] + '_pu.h5'
    file_name_no_pu = 'noPU/' + f1.split('/')[11].split('.')[0] + '_no_pu.h5'

    h5f_pu = h5py.File(file_name_pu, 'w')
    h5f_no_pu = h5py.File(file_name_no_pu, 'w')

    h5f_pu.create_dataset('image', data=pics[np.where(pus==200)])
    h5f_pu.create_dataset('true_energy', data=trues[np.where(pus==200)])
    
    h5f_no_pu.create_dataset('image', data=pics[np.where(pus==0)])
    h5f_no_pu.create_dataset('true_energy', data=trues[np.where(pus==0)])

    h5f_pu.close()
    h5f_no_pu.close()


def main():
    d = '/eos/cms/store/cmst3/group/hgcal/CMG_studies/Production/FlatRandomPtGunProducer_jkiesele_PDGid11_id13_id211_id22_x8_Pt2.0To100_PU200_20170914/DeepHGCalData_merged_with_0PU/'

    root_files = glob.glob(d + '*.root')
    #root_files.remove('/eos/cms/store/cmst3/group/hgcal/CMG_studies/Production/convertedH_FlatRandomPtGunProducer_jkiesele_PDGid11_id13_id211_id22_x8_Pt2.0To100_PU200_20170914/fourth/train_fourth/partGun_PDGid11_x160_Pt2.0To100.0_NTUP_199.root')

    for f in root_files:
        pics, trues, pus = convert(f)
        toH5(f, pics, trues, pus)
        
    print("Done!")


if __name__ == "__main__":
    if not os.path.exists(os.getcwd()+"/PU/"):
        os.makedirs(os.getcwd()+"/PU/")

    if not os.path.exists(os.getcwd()+"/noPU/"):
        os.makedirs(os.getcwd()+"/noPU/")
    main()
