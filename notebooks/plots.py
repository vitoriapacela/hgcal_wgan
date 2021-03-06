import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.stats import skew, kurtosis, wasserstein_distance
import scipy as sp
from matplotlib.colors import LogNorm, Normalize

def read_h5(h5file1):
    '''
    Reads HDF5 containing the loss and returns the losses as numpy arrays.
    :param h5file1: name of the file
    :type h5file1: str
    :return: 4 numpy arrays
    '''
    #import h5py
    f = h5py.File(h5file1, "r")
    discriminator1 = np.asarray(f["discriminator"])
    discriminator_fake1 = np.asarray(f["discriminator_fake"])
    discriminator_real1 = np.asarray(f["discriminator_real"])
    generator1 = np.asarray(f["generator"])
    f.close()
    return discriminator1, discriminator_fake1, discriminator_real1, generator1


def plotLoss(data, title="", tag=""):
    '''
    Plots individual loss (eg. either generator or discriminator).
    :param data: loss data as numpy array
    :param title: (str) title of the plot
    :param tag: (str) optional tag to put in the title
    :return:
    '''
    fig = plt.figure()
    plt.plot(data[:, 0])
    plt.title(tag + title + " loss", size=16)
    plt.xlabel("Step", size=16)
    plt.ylabel("Total loss", size=16)
    # plt.xlim(-50, 1000)
    # plt.savefig(title + "_losses.png")
    

def losses(h5file):
    '''
    Plots multiple losses for a model.
    :param h5file: (str) name of the file containing loss arrays
    :return:
    '''
    discriminator, discriminator_fake, discriminator_real, generator = read_h5(h5file)

    plotLoss(1 - discriminator, "Discriminator")
    plotLoss(1 - discriminator_real, "Discriminator_real")
    plotLoss(1 - discriminator_fake, "Discriminator_fake")
    plotLoss(1 - generator, "Generator")


def plotLossVal(loss, val_loss, title, tag=""):
    '''
    Plots training loss and validation loss together.
    :param loss: training loss as numpy array
    :param val_loss: validation loss as numpy array
    :param title: (str) title of the plot
    :param tag: (str) optional tag to put in the title
    :return:
    '''
    #import h5py
    fig = plt.figure()
    plt.plot(loss[:, 0], label="train", color='blue', alpha=0.4)
    plt.plot(val_loss[:, 0], label="validation", color='red', alpha=0.4)
    plt.title(tag + title + " loss", size=16)
    plt.xlabel("Step", size=16)
    plt.ylabel("W loss", size=16)
    plt.legend(prop={'size': 15})
    # plt.xlim(-50, 1000)
    # plt.savefig(title + "_losses_val.png")
    plt.show()
    
    
def plotWassLoss(data, title="", tag=""):
    '''
    Plots individual loss (eg. either generator or discriminator).
    :param data: loss data as numpy array
    :param title: (str) title of the plot
    :param tag: (str) optional tag to put in the title
    :return:
    '''
    fig = plt.figure()
    plt.plot(data[:, 1])
    plt.title(tag + title + " loss", size=16)
    plt.xlabel("Step", size=16)
    plt.ylabel("W loss", size=16)
    # plt.xlim(-50, 1000)
    # plt.savefig(title + "_losses.png")
    
    
def plotEnergyLoss(data, title="", tag=""):
    '''
    Plots individual loss (eg. either generator or discriminator).
    :param data: loss data as numpy array
    :param title: (str) title of the plot
    :param tag: (str) optional tag to put in the title
    :return:
    '''
    fig = plt.figure()
    plt.plot(data[:, 2])
    plt.title(tag + title + " loss", size=16)
    plt.xlabel("Step", size=16)
    plt.ylabel("MSE", size=16)
    # plt.xlim(-50, 1000)
    # plt.savefig(title + "_losses.png")


def lossesVal(h5file, h5file_val):
    '''
    Plots training and validation losses together for multiple losses
    :param h5file: (str) name of the file containing training loss arrays
    :param h5file_val: (str) name of the file containing validation loss arrays
    :return:
    '''
    discriminator, discriminator_fake, discriminator_real, generator = read_h5(h5file)
    val_discriminator, val_discriminator_fake, val_discriminator_real, val_generator = read_h5(h5file_val)

    plotLossVal(1 - discriminator, 1 - val_discriminator, "Discriminator")
    plotLossVal(1 - discriminator_real, 1 - val_discriminator_real, title="Discriminator_real")
    plotLossVal(1 - discriminator_fake, 1 - val_discriminator_fake, title="Discriminator_fake")
    # plotLoss(generator, "generator")


def plotLoss2(loss1, loss2, title, tag=""):
    fig = plt.figure()
    plt.plot(loss1[:, 0], label="first", color='blue', alpha=0.4)
    plt.plot(loss2[:, 0], label="current", color='red', alpha=0.4)
    plt.title(tag + title + " loss", size=16)
    plt.xlabel("Step", size=16)
    plt.ylabel("W loss", size=16)
    plt.legend(prop={'size': 15})
    plt.xlim(-50, 2000)
    # plt.savefig("compare_bs_32_128_" + title + "_losses.png")
    plt.show()


def losses2(h5file1, h5file2):
    # (str) h5file: name of the file containing the loss arrays
    discriminator1, discriminator_fake1, discriminator_real1, generator1 = read_h5(h5file1)
    discriminator2, discriminator_fake2, discriminator_real2, generator2 = read_h5(h5file2)

    plotLoss2(1 - discriminator1, 1 - discriminator2, "Discriminator")
    plotLoss2(1 - discriminator_real1, 1 - discriminator_real2, "Discriminator_real")
    plotLoss2(1 - discriminator_fake1, 1 - discriminator_fake2, "Discriminator_fake")
    plotLoss2(1 - generator1, 1 - generator2, "Generator")


def plotLoss3(loss1, loss2, loss3, title, tag=""):
    fig = plt.figure()
    plt.plot(loss1[:, 0], label="bs=32, lr=0.00005", color='blue', alpha=0.4)
    plt.plot(loss2[:, 0], label="bs=32, lr=0.005", color='red', alpha=0.4)
    plt.plot(loss3[:, 0], label="bs=128, lr=0.00005", color='green', alpha=0.4)
    plt.title(tag + title + " loss", size=16)
    plt.xlabel("Step", size=16)
    plt.ylabel("W loss", size=16)
    plt.legend(prop={'size': 15})
    plt.xlim(-50, 1000)
    # plt.savefig("compare_bs_32_128_" + title + "_losses.png")
    plt.show()


def losses3(h5file1, h5file2, h5file3):
    # (str) h5file: name of the file containing the loss arrays
    discriminator1, discriminator_fake1, discriminator_real1, generator1 = read_h5(h5file1)
    discriminator2, discriminator_fake2, discriminator_real2, generator2 = read_h5(h5file2)
    discriminator3, discriminator_fake3, discriminator_real3, generator3 = read_h5(h5file3)

    plotLoss3(1 - discriminator1, 1 - discriminator2, 1 - discriminator3, "Discriminator")
    plotLoss3(1 - discriminator_real1, 1 - discriminator_real2, 1 - discriminator_real3, "Discriminator_real")
    plotLoss3(1 - discriminator_fake1, 1 - discriminator_fake2, 1 - discriminator_fake3, "Discriminator_fake")
    plotLoss3(1 - generator1, 1 - generator2, 1 - generator2, "Generator")


def plotRealFake(real, fake, title, tag=""):
    fig = plt.figure()
    plt.plot(real[:, 0], label="real", color='blue', alpha=0.5)
    plt.plot(fake[:, 0], label="fake", color='red', alpha=0.5)
    plt.title(tag + title + " loss", size=16)
    plt.xlabel("Step", size=16)
    plt.ylabel("W loss", size=16)
    plt.legend(prop={'size': 15})
    # plt.savefig(title + "_losses_real_fake.png")
    # plt.xlim(-50, 2000)
    # plt.ylim(0.5, 1.5)
    plt.show()


def lossesRealFake(h5file1):
    # (str) h5file: name of the file containing the loss arrays
    discriminator1, discriminator_fake1, discriminator_real1, generator1 = read_h5(h5file1)

    plotRealFake(1 - discriminator_real1, 1 - discriminator_fake1, "Discriminator")

def loadModel(name, weights=False):
    '''
    Loads models from json file.
    :parameter name: name of the json file.
    :type name: str
    :parameter weights: whether or not to load the weights.
    :type weights: bool
    :return: loaded model.
    '''
    from keras.models import model_from_json
    json_file = open('%s' % name, 'r')
    loaded = json_file.read()
    json_file.close()

    model = model_from_json(loaded)

    # load weights into new model
    if weights == True:
        model.load_weights('%s.h5' % name)
    # print(model.summary())

    #print("Loaded model from disk")
    return model


def getMetric(all_g_weight, gen_model):
    '''
    Generate predictions from models and saves the moments of the distribution.
    :param all_g_weight: (str) path to the directory containing all the weight HDF5 files from the generator.
    :param gen_model: (str) path to json file containing the saved generator model.
    :return: 6 numpy arrays
    '''
    g = loadModel(gen_model)

    means = []
    stds = []
    epochs = []
    variances = []
    skews = []
    kurtoses = []

    latent_space=100
    batch_size=128

    noise = np.random.normal(0, 1, (batch_size, latent_space))
    for w in glob.glob(all_g_weight):
        epoch = w.split('/')[-1].split('_')[3]

        g.load_weights(w)
        generated_images = g.predict(noise)
        # generated_images = generated_images.squeeze()

        means.append(np.mean(generated_images))
        stds.append(np.std(generated_images))
        epochs.append(int(epoch))
        variances.append(np.var(generated_images))
        skews.append(skew(generated_images, axis=None))
        kurtoses.append(kurtosis(generated_images, axis=None))

    return means, stds, epochs, variances, skews, kurtoses


def sortMeans(epochs, means):
    '''
    Sorts the array containing the means according the epoch.
    :param epochs: numpy array containing the epochs
    :param means: numpy array containing the means
    :return:
    '''
    epoch_mean = np.array([np.asarray(epochs), np.asarray(means)])
    epoch_mean_sorted = (epoch_mean.T)[np.argsort(epoch_mean[0])]
    return epoch_mean_sorted


def plotMean(epoch_mean_sorted):
    plt.scatter(epoch_mean_sorted[:,0], epoch_mean_sorted[:,1], alpha=0.6, color='blue')
    plt.title("Mean energy generated", size=16)
    plt.xlabel("Step", size=16)
    plt.ylabel("$\mu$ (GeV)", size=16)
    #plt.xlim(-50, 1000)
    ## ACTIVATE YLIM
    #plt.ylim(0, 0.05)
    #plt.savefig("means.png")
    plt.errorbar(epoch_mean_sorted[:,0], epoch_mean_sorted[:,1], yerr=np.std(epoch_mean_sorted[:,1])/np.sqrt(len(epoch_mean_sorted[:,1])), color='orange', alpha = 0.3, fmt='o')


def plotMeans(epoch_mean_sorted1, epoch_mean_sorted2):
    plt.scatter(epoch_mean_sorted1[:,0], epoch_mean_sorted1[:,1], label="first")
    plt.scatter(epoch_mean_sorted2[:,0], epoch_mean_sorted2[:,1], label="second")
    plt.title("Mean energy generated", size=16)
    plt.xlabel("Step", size=16)
    plt.ylabel("$\mu$ (GeV)", size=16)
    plt.xlim(-50, 1000)
    #plt.ylim(0, 1.5)
    plt.legend()
    #plt.savefig("means.png")
    #plt.errorbar(epoch_mean_sorted[:,0], epoch_mean_sorted[:,1], yerr=np.std(epoch_mean_sorted[:,1])/np.sqrt(len(epoch_mean_sorted[:,1])), color='grey', alpha = 0.5, fmt='o')



def plotStd(epochs, stds):
    plt.scatter(epochs, stds, alpha=0.5, color='green')
    plt.title("Standard deviation of energy generated", size=16)
    plt.xlabel("Step", size=16)
    plt.ylabel("$\sigma$ (GeV)", size=16)
    #plt.xlim(-50, 1000)
    plt.ylim(0, 0.5)
    #plt.savefig("stds.png")


def plotStds(epochs1, stds1, epochs2, stds2):
    plt.scatter(epochs1, stds1, label="first")
    plt.scatter(epochs2, stds2, label="second")
    plt.title("Standard deviation of energy generated", size=16)
    plt.xlabel("Step", size=16)
    plt.ylabel("$\sigma$ (GeV)", size=16)
    plt.xlim(-50, 1000)
    #plt.ylim(0, 1)
    plt.legend()
    #plt.savefig("stds.png")


def plotVar(epochs, variances):
    plt.scatter(epochs, variances, alpha=0.5, color='green')
    plt.title("Variance of sample generated per epoch", size=16)
    plt.xlabel("Step", size=16)
    plt.ylabel("Var", size=16)
    #plt.xlim(-50, 1000)
    #plt.ylim(0, 0.5)
    #plt.savefig("stds.png")


def plotSkew(epochs, skews):
    plt.scatter(epochs, skews, alpha=0.5, color='green')
    plt.title("Skewness", size=16)
    plt.xlabel("Step", size=16)
    plt.ylabel(r"$\gamma$", size=16)
    #plt.xlim(-50, 1000)
    #plt.ylim(0, 0.5)
    #plt.savefig("stds.png")

    
def plotKurtosis(epochs, kurtoses):
    plt.scatter(epochs, kurtoses, alpha=0.5, color='green')
    plt.title("Kurtosis", size=16)
    plt.xlabel("Step", size=16)
    plt.ylabel(r"$\kappa$", size=16)
    #plt.xlim(-50, 1000)
    #plt.ylim(0, 0.5)
    #plt.savefig("stds.png")

    
def plotHistogram(real_sum, generated_sum, epoch='', bins=7):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    na, bina, _ = ax.hist(real_sum, alpha=0.3, bins=bins, label='real sum', color='blue')
    nb, binb, _ = ax.hist(generated_sum, alpha=0.3, bins=bins, label='generated sum', color='red')
    plt.legend(prop={'size': 15})
    plt.title("Total energy, step " + str(epoch), size=16)
    ax.set_xlabel('Energy (GeV)', size=16)
    ax.set_ylabel('n events', size=16)
    #plt.show()
    #plt.savefig('images_sum/pred_%s.png'%epoch)
    return [(na, bina), (nb, binb)]

    
def sortByStep(e):
    step = int(e.split('/')[-1].split('_')[3])
    return step


def plotSumHist(inp_sum, all_g_weight, gen_model):
    g = loadModel(gen_model)
    
    n_samples = 200
    latent_space=100
    noise = np.random.normal(0, 1, (n_samples, latent_space))

    weight_files = glob.glob(all_g_weight)
    weight_files.sort(key=sortByStep)
    
    for w in weight_files:
        step = w.split('/')[-1].split('_')[3]
        g.load_weights(w)
        generated_images = g.predict(noise)
        generated_images = generated_images.squeeze()
        gen_sum = np.sum(generated_images, axis=(1, 2, 3))
        plotHistogram(inp_sum, gen_sum, step, bins = 20)


def plotX(X, generated_images, step='', n_samples=200):
    # Plot sum over X axis:
    x = np.arange(16)
    y_real = np.sum(X[0:n_samples-1], axis=(0, 2, 3))
    y_fake = np.sum(generated_images, axis=(0, 2, 3))
    plt.plot(x, y_real, color="green", label="real", alpha=0.9)
    plt.plot(x, y_fake, color="orange", label="generated", alpha=0.9)
    plt.xlabel("Position over the x axis", size=16)
    plt.ylabel("Energy sum (GeV)", size=16)
    #plt.title("x axis", size=16)
    plt.title("Energy x axis, step " + str(step), size=16)
    plt.legend()
    plt.show()


def plotY(X, generated_images, step='', n_samples=200):
    # Plot sum over Y axis:
    x = np.arange(16)
    y_real = np.sum(X[0:n_samples-1], axis=(0, 1, 3))
    y_fake = np.sum(generated_images, axis=(0, 1, 3))
    plt.plot(x, y_real, color="green", label="real", alpha=0.9)
    plt.plot(x, y_fake, color="orange", label="generated", alpha=0.9)
    plt.xlabel("Position over the y axis", size=16)
    plt.ylabel("Energy sum (GeV)", size=16)
    #plt.title("y axis", size=16)
    plt.title("Energy y axis, step " + str(step), size=16)
    plt.legend()
    plt.show()


def plotZ(X, generated_images, step = '', n_samples=200):
    # Plot sum over Z axis:
    x = np.arange(55)
    y_real = np.sum(X[0:n_samples-1], axis=(0, 1, 2))
    y_fake = np.sum(generated_images, axis=(0, 1, 2))
    plt.plot(x, y_real, color="green", label="real", alpha=0.8)
    plt.plot(x, y_fake, color="orange", label="generated", alpha=0.8)
    plt.xlabel("Position over the z axis", size=16)
    plt.ylabel("Energy sum (GeV)", size=16)
    #plt.title("z axis", size=16)
    plt.title("Energy z axis, step " + str(step), size=16)
    plt.legend()
    plt.show()


def plotAxes(inp, all_g_weight, gen_model, n_samples=200):
    g = loadModel(gen_model)
    latent_space=100
    noise = np.random.normal(0, 1, (n_samples, latent_space))
    
    weight_files = glob.glob(all_g_weight)
    weight_files.sort(key=sortByStep)
    
    for w in weight_files:
        step = w.split('/')[-1].split('_')[3]
        g.load_weights(w)
        generated_images = g.predict(noise)

        plotX(inp, generated_images, step)
        plotY(inp, generated_images, step)
        plotZ(inp, generated_images, step)
        
        
def saveModel(model, name="gan"):
    '''
    Saves model as json file..
    :parameter model: model to be saved.
    :parameter name: name of the model to be saved.
    :type name: str
    :return: saved model.
    '''
    model_name = name
    model.summary()
    #model.save_weights('%s.h5' % model_name, overwrite=True)
    model_json = model.to_json()
    with open("%s.json" % model_name, "w") as json_file:
        json_file.write(model_json)
        
        
def saveLosses(name, discr_real, discr_fake, discr, gen):
    '''
    Saves true energy and prediction energy arrays into an HDF5 file.
    :parameter name: name of the file to be saved.
    :type name: str.
    '''
    new_file = h5py.File(name + "_losses.h5", 'a')
    new_file.create_dataset("discriminator_real", data=discr_real)
    new_file.create_dataset("discriminator_fake", data=discr_fake)
    new_file.create_dataset("discriminator", data=discr)
    new_file.create_dataset("generator", data=gen)
    new_file.close()
    
    
def KL(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)

    a = a/np.sum(a)
    b = b/np.sum(b)
    
    return np.sum(np.where(a != 0, a * np.log(a / b), 0))


def jsd(p, q, base=np.e):
    '''
        Implementation of pairwise `jsd` based on  
        https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
    '''
    ## convert to np.array
    p, q = np.asarray(p), np.asarray(q)
    ## normalize p, q to probabilities
    p, q = p/p.sum(), q/q.sum()
    m = 1./2*(p + q)
    return sp.stats.entropy(p,m, base=base)/2. +  sp.stats.entropy(q, m, base=base)/2.


def getDists(all_g_weight, gen_model, file_name, latent_space=100, n_samples = 2000):
    f = h5py.File(file_name, 'r')
    X = np.asarray(f['image'])
    np.random.shuffle(X)
    
    g = loadModel(gen_model)
    noise = np.random.normal(0, 1, (n_samples, latent_space))
    X_n = X[0:n_samples]
    
    w_dists = wDist(all_g_weight, g, X_n, noise)
    kl_divs = kls(all_g_weight, g, X_n, noise)
    js_divs = js(all_g_weight, g, X_n, noise)
    
    return w_dists, kl_divs, js_divs


def wDist(all_g_weight, g, X_n, noise):
    wx = []
    wy = []
    wz = []
    epochs = []

    for w in glob.glob(all_g_weight):
        epoch = w.split('/')[-1].split('_')[3]
        epoch = int(epoch)

        g.load_weights(w)
        generated_images = g.predict(noise)

        gen_im_x = np.sum(generated_images, axis=(2, 3))
        gen_im_x = np.mean(gen_im_x, axis=0)

        real_img_x = np.sum(X_n, axis=(2, 3))
        real_img_x = np.mean(real_img_x, axis=0)

        gen_im_y = np.sum(generated_images, axis=(1, 3))
        gen_im_y = np.mean(gen_im_y, axis=0)

        real_img_y = np.sum(X_n, axis=(1, 3))
        real_img_y = np.mean(real_img_y, axis=0)

        gen_im_z = np.sum(generated_images, axis=(1, 2))
        gen_im_z = np.mean(gen_im_z, axis=0)

        real_img_z = np.sum(X_n, axis=(1, 2))
        real_img_z = np.mean(real_img_z, axis=0)
        
        real_img_x_norm = real_img_x/np.sum(real_img_x)
        gen_img_x_norm = gen_im_x/np.sum(gen_im_x)
        
        real_img_y_norm = real_img_y/np.sum(real_img_y)
        gen_img_y_norm = gen_im_y/np.sum(gen_im_y)
        
        real_img_z_norm = real_img_z/np.sum(real_img_z)
        gen_img_z_norm = gen_im_z/np.sum(gen_im_z)

        wx.append(wasserstein_distance(real_img_x_norm, gen_img_x_norm))
        wy.append(wasserstein_distance(real_img_y_norm, gen_img_y_norm))
        wz.append(wasserstein_distance(real_img_z_norm, gen_img_z_norm))
        epochs.append(epoch)
        
    epoch_w = np.array([np.asarray(epochs), np.asarray(wx), np.asarray(wy), np.asarray(wz)])
    epoch_w_sorted = (epoch_w.T)[np.argsort(epoch_w[0])]
    return epoch_w_sorted


def kls(all_g_weight, g, X_n, noise):
    wx = []
    wy = []
    wz = []
    epochs = []

    for w in glob.glob(all_g_weight):
        epoch = w.split('/')[-1].split('_')[3]
        epoch = int(epoch)

        g.load_weights(w)
        generated_images = g.predict(noise)

        gen_im_x = np.sum(generated_images, axis=(2, 3))
        gen_im_x = np.mean(gen_im_x, axis=0)

        real_img_x = np.sum(X_n, axis=(2, 3))
        real_img_x = np.mean(real_img_x, axis=0)

        gen_im_y = np.sum(generated_images, axis=(1, 3))
        gen_im_y = np.mean(gen_im_y, axis=0)

        real_img_y = np.sum(X_n, axis=(1, 3))
        real_img_y = np.mean(real_img_y, axis=0)

        gen_im_z = np.sum(generated_images, axis=(1, 2))
        gen_im_z = np.mean(gen_im_z, axis=0)

        real_img_z = np.sum(X_n, axis=(1, 2))
        real_img_z = np.mean(real_img_z, axis=0)
        
        real_img_x_norm = real_img_x/np.sum(real_img_x)
        gen_img_x_norm = gen_im_x/np.sum(gen_im_x)
        
        real_img_y_norm = real_img_y/np.sum(real_img_y)
        gen_img_y_norm = gen_im_y/np.sum(gen_im_y)
        
        real_img_z_norm = real_img_z/np.sum(real_img_z)
        gen_img_z_norm = gen_im_z/np.sum(gen_im_z)

        wx.append(KL(real_img_x_norm, gen_img_x_norm))
        wy.append(KL(real_img_y_norm, gen_img_y_norm))
        wz.append(KL(real_img_z_norm, gen_img_z_norm))
        epochs.append(epoch)
        
    epoch_w = np.array([np.asarray(epochs), np.asarray(wx), np.asarray(wy), np.asarray(wz)])
    epoch_w_sorted = (epoch_w.T)[np.argsort(epoch_w[0])]
    return epoch_w_sorted


def js(all_g_weight, g, X_n, noise):
    wx = []
    wy = []
    wz = []
    epochs = []

    for w in glob.glob(all_g_weight):
        epoch = w.split('/')[-1].split('_')[3]
        epoch = int(epoch)

        g.load_weights(w)
        generated_images = g.predict(noise)

        gen_im_x = np.sum(generated_images, axis=(2, 3))
        gen_im_x = np.mean(gen_im_x, axis=0)

        real_img_x = np.sum(X_n, axis=(2, 3))
        real_img_x = np.mean(real_img_x, axis=0)

        gen_im_y = np.sum(generated_images, axis=(1, 3))
        gen_im_y = np.mean(gen_im_y, axis=0)

        real_img_y = np.sum(X_n, axis=(1, 3))
        real_img_y = np.mean(real_img_y, axis=0)

        gen_im_z = np.sum(generated_images, axis=(1, 2))
        gen_im_z = np.mean(gen_im_z, axis=0)

        real_img_z = np.sum(X_n, axis=(1, 2))
        real_img_z = np.mean(real_img_z, axis=0)

        real_img_x_norm = real_img_x/np.sum(real_img_x)
        gen_img_x_norm = gen_im_x/np.sum(gen_im_x)
        
        real_img_y_norm = real_img_y/np.sum(real_img_y)
        gen_img_y_norm = gen_im_y/np.sum(gen_im_y)
        
        real_img_z_norm = real_img_z/np.sum(real_img_z)
        gen_img_z_norm = gen_im_z/np.sum(gen_im_z)

        wx.append(jsd(real_img_x_norm, gen_img_x_norm))
        wy.append(jsd(real_img_y_norm, gen_img_y_norm))
        wz.append(jsd(real_img_z_norm, gen_img_z_norm))
        epochs.append(epoch)
        
    epoch_w = np.array([np.asarray(epochs), np.asarray(wx), np.asarray(wy), np.asarray(wz)])
    epoch_w_sorted = (epoch_w.T)[np.argsort(epoch_w[0])]
    return epoch_w_sorted


def plotDists(w_dist, kls, jss):
    plt.figure(figsize=(6, 6))

    plt.errorbar(w_dist[:, 0], np.mean(w_dist[:, [1, 2, 3]], axis=1), yerr=np.std(w_dist[:, [1, 2, 3]], axis=1)/np.sqrt(len(w_dist[:, 1])), color='orange', alpha = 0.3, fmt='o', label='WD')
    plt.errorbar(kls[:, 0], np.mean(kls[:, [1, 2, 3]], axis=1), yerr=np.std(kls[:, [1, 2, 3]], axis=1)/np.sqrt(len(kls[:, 1])), color='red', alpha = 0.3, fmt='o', label='KL')
    plt.errorbar(jss[:, 0], np.mean(jss[:, [1, 2, 3]], axis=1), yerr=np.std(jss[:, [1, 2, 3]], axis=1)/np.sqrt(len(jss[:, 1])), alpha = 0.3, fmt='o', color = 'blue', label='JS')

    plt.xlabel("Steps", size=16)
    plt.ylabel(r"$\mu_{div}$", size=16)
    plt.legend(prop={'size': 16})
    plt.yscale('log')
    
    
def plot_wdist_xyz(w_dist):
    plt.figure(figsize=(6, 6))

    plt.scatter(w_dist[:, 0], w_dist[:, 1], color='red', alpha=0.3, label='x')
    plt.scatter(w_dist[:, 0], w_dist[:, 2], color='blue', alpha=0.3, label='y')
    plt.scatter(w_dist[:, 0], w_dist[:, 3], color='green', alpha=0.3, label='z')

    plt.xlabel("Steps", size=16)
    plt.ylabel("Wasserstein distance", size=16)
    plt.legend(prop={'size': 16})
    plt.yscale('log')
    
    
def compare_avg_wDist(w_dist1, w_dist2):
    plt.figure(figsize=(6, 6))

    plt.errorbar(w_dist1[:, 0], np.mean(w_dist1[:, [1, 2, 3]], axis=1), yerr=np.std(w_dist1[:, [1, 2, 3]], axis=1)/np.sqrt(len(w_dist1[:, 1])), color='orange', alpha = 0.3, fmt='o', label='w/ reg')
    plt.errorbar(w_dist2[:, 0], np.mean(w_dist2[:, [1, 2, 3]], axis=1), yerr=np.std(w_dist2[:, [1, 2, 3]], axis=1)/np.sqrt(len(w_dist2[:, 1])), color='green', alpha = 0.3, fmt='o', label='w/o reg')

    plt.xlabel("Steps", size=16)
    plt.ylabel(r"$\mu_{div}$", size=16)
    plt.legend(prop={'size': 16})
    plt.yscale('log')
    #plt.xlim(0, 4000)
    
    
def weigh_hist_x(X, dim_size=10, log=False, color='purple'):
    '''
    Plots a histogram of the energy respective to the calorimeter cell id.
    '''
    xx = np.arange(dim_size)
    xx = np.tile(xx, dim_size)
    # Array xx has n x n entries for the 100 cells in the x-y projection, n = 10. The entries are in the range (0, 9).
    
    X_red = np.sum(X, axis=(3))
    X_red = np.mean(X_red, axis=0)
    X_red = X_red.reshape(dim_size**2)
    # Array X_red has the energy for each cell in the xx array.
    
    # Then we plot a weighted histogram, in which X_red are the weights for the cells in xx.
    # plt.hist does not accept matrixes, hence the arrays were flattened.
    plt.hist(xx, dim_size, weights=X_red, log=log, color=color)
    plt.xlabel('x', size=16)
    plt.ylabel('Total energy (GeV)', size=16)
    plt.title('Energy in x-axis projection', size=16)
    
    
def weigh_hist_y(X, dim_size=10, log=False, color='purple'):
    xx = np.arange(dim_size)
    xx = np.tile(xx, dim_size)
    # Array xx has n x n entries for the 100 cells in the x-y projection, n = 10. The entries are in the range (0, 9).
    
    X_red = np.sum(X, axis=(3))
    X_red = np.mean(X_red, axis=0)
    X_red = X_red.reshape(dim_size**2)
    # Array X_red has the energy for each cell in the xx array.
    
    # Then we plot a weighted histogram, in which X_red are the weights for the cells in xx.
    # plt.hist does not accept matrixes, hence the arrays were flattened.
    plt.hist(xx, dim_size, weights=X_red.T, log=log, color=color)
    plt.xlabel('y', size=16)
    plt.ylabel('Total energy (GeV)', size=16)
    plt.title('Energy in y-axis projection', size=16)
    
    
def weigh_hist_z(X, dim_size=30, log=False, color='purple'):
    xx = np.arange(dim_size)
    xx = np.tile(xx, 10)
    # Array xx has 30 x 10 entries for the 300 cells in the x-z projection. The entries are in the range (0, 29).
    
    X_red = np.sum(X, axis=(2))
    X_red = np.mean(X_red, axis=0)
    X_red = X_red.flatten()
    # Array X_red has the energy for each cell in the xx array.
    
    # Then we plot a weighted histogram, in which X_red are the weights for the cells in xx.
    # plt.hist does not accept matrixes, hence the arrays were flattened.
    plt.hist(xx, dim_size, weights=X_red, log=log, color=color)
    plt.xlabel('z', size=16)
    plt.ylabel('Total energy (GeV)', size=16)
    plt.title('Energy in z-axis projection', size=16)
    
    
def plt2d(X):
    X_red = np.sum(X, axis=(3))
    X_red = np.mean(X_red, axis=0)
    plt.imshow(X_red.T, origin = 'lower')
    plt.colorbar()
    plt.title("Total energy (GeV) in x-y projection", size=16)
    plt.xlabel('x', size=16)
    plt.ylabel('y', size=16)
    
    
def plt2d_log(X):    
    X_red = np.sum(X, axis=(3))
    X_red = np.mean(X_red, axis=0)
    
    plt.imshow(X_red.T, origin = 'lower', norm=LogNorm(vmin=0.01, vmax=1))
    plt.colorbar()
    plt.title("Total energy (GeV) in x-y projection", size=16)
    plt.xlabel('x', size=16)
    plt.ylabel('y', size=16)