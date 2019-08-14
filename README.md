# hgcal_wgan

WGANs for HGCAL data at CMS.


## Dependencies:
* [`NumPy`](http://www.numpy.org/)
* [`matplotlib`](https://matplotlib.org/)
* [`h5py`](http://www.h5py.org/)
* `setGPU`
* `gpustat`
* [`tensorflow-gpu`](https://www.tensorflow.org/)
* [`keras`](https://keras.io/) (v. >= 1.2.0)
* [`root_numpy`](https://github.com/scikit-hep/root_numpy)


## Structure
* `env` contains scripts to set up the environment at SF's GPUs.
* `preprocessing` contains tools to preprocess ROOT files and convert them to HDF5.
* `gans` contains GAN implementations for training. The best so far is `wgan_conv2d_regression.py`.
* `notebooks` contains Jupyter Notebooks used for evaluation.
