import h5py
import glob
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

truth = np.array([])
raw = np.array([])

src = '/afs/cern.ch/work/v/vbarinpa/hgcal_wgan/preprocessing/noPU/*.h5'
dest = '/afs/cern.ch/work/v/vbarinpa/hgcal_wgan/preprocessing/all_noPU.h5'

for fileName in glob.glob(src):
    print(fileName)
    myFile = h5py.File(fileName)
    myraw = myFile.get('image') 
    mytruth = myFile.get('true_energy')
    
    raw = np.concatenate((raw, myraw), axis=0) if raw.size else myraw
    truth = np.concatenate((truth ,mytruth), axis=0) if truth.size else mytruth

print('array')
    
fileOut = h5py.File(dest, 'w')

print('file created')

fileOut.create_dataset('image', data=raw,compression='gzip')
fileOut.create_dataset('true_energy', data=truth,compression='gzip')

print('done')

fileOut.close()
