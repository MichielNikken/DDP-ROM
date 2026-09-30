import os

import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio
from IPython.display import clear_output as clc
from IPython.display import display

from sklearn.utils.extmath import randomized_svd

mae = lambda datatrue, datapred: (datatrue - datapred).abs().mean()
mse = lambda datatrue, datapred: (datatrue - datapred).pow(2).sum(axis=-1).mean()
mre = lambda datatrue, datapred: (
            (datatrue - datapred).pow(2).sum(axis=-1).sqrt() / (datatrue).pow(2).sum(axis=-1).sqrt()).mean()
num2p = lambda prob: ("%.2f" % (100 * prob)) + "%"

from sklearn.preprocessing import MinMaxScaler

def compute_svd_trainset(train_loader, train_loader_svd, test_loader, test_loader_svd, valid_loader, valid_loader_svd,
                            trunc_dim, ntrain, nvalid, ntest, timesteps_vec, nvelocity, ntimes, Vtest, nparams, exp='flowpastobject'):

    train_data = train_loader.get_all_samples()
    Vxtrain = train_data['vx_snapshots']
    Vytrain = train_data['vy_snapshots']
    MUtrain = train_data['mus']

    del train_data

    # Ux, Sx, Wx = randomized_svd(Vxtrain, n_components=trunc_dim // 2)
    # Uy, Sy, Wy = randomized_svd(Vytrain, n_components=trunc_dim // 2)

    _, _, Wx = randomized_svd(Vxtrain, n_components=trunc_dim)
    _, _, Wy = randomized_svd(Vytrain, n_components=trunc_dim)

    valid_data = valid_loader.get_all_samples()
    Vxvalid = valid_data['vx_snapshots']
    Vyvalid = valid_data['vy_snapshots']
    MUvalid = valid_data['mus']

    del valid_data

    test_data = test_loader.get_all_samples()
    Vxtest = test_data['vx_snapshots']
    Vytest = test_data['vy_snapshots']
    MUtest = test_data['mus']

    del test_data

    Vxtrain_POD = Vxtrain @ Wx.transpose()
    Vxvalid_POD = Vxvalid @ Wx.transpose()
    Vxtest_POD = Vxtest @ Wx.transpose()
    #Vxtrain_reconstructed = Ux @ np.diag(Sx) @ Wx
    #Vxvalid_reconstructed = Vxvalid @ Wx.transpose() @ Wx
    Vxtest_reconstructed = Vxtest @ Wx.transpose() @ Wx

    Vytrain_POD = Vytrain @ Wy.transpose()
    Vyvalid_POD = Vyvalid @ Wy.transpose()
    Vytest_POD = Vytest @ Wy.transpose()
    #Vytrain_reconstructed = Uy @ np.diag(Sy) @ Wy
    #Vyvalid_reconstructed = Vyvalid @ Wy.transpose() @ Wy
    Vytest_reconstructed = Vytest @ Wy.transpose() @ Wy

    # SCALING
    scalerVx = MinMaxScaler()
    scalerVx = scalerVx.fit(Vxtrain_POD)
    # Vxtrain_POD = scalerVx.transform(Vxtrain_POD)
    # Vxvalid_POD = scalerVx.transform(Vxvalid_POD)
    # Vxtest_POD = scalerVx.transform(Vxtest_POD)
    #
    scalerVy = MinMaxScaler()
    scalerVy = scalerVy.fit(Vytrain_POD)
    # Vytrain_POD = scalerVy.transform(Vytrain_POD)
    # Vyvalid_POD = scalerVy.transform(Vyvalid_POD)
    # Vytest_POD = scalerVy.transform(Vytest_POD)

    # RESHAPE MATRICES
    Vxtrain = torch.from_numpy(Vxtrain.reshape(ntrain, ntimes, nvelocity // 2))
    Vxvalid = torch.from_numpy(Vxvalid.reshape(nvalid, ntimes, nvelocity // 2))
    Vxtest = torch.from_numpy(Vxtest.reshape(ntest, ntimes, nvelocity // 2))
    Vxtrain_POD = torch.from_numpy(Vxtrain_POD.reshape(ntrain, ntimes, trunc_dim))
    Vxvalid_POD = torch.from_numpy(Vxvalid_POD.reshape(nvalid, ntimes, trunc_dim))
    Vxtest_POD = torch.from_numpy(Vxtest_POD.reshape(ntest, ntimes, trunc_dim))

    Vytrain = torch.from_numpy(Vytrain.reshape(ntrain, ntimes, nvelocity // 2))
    Vyvalid = torch.from_numpy(Vyvalid.reshape(nvalid, ntimes, nvelocity // 2))
    Vytest = torch.from_numpy(Vytest.reshape(ntest, ntimes, nvelocity // 2))
    Vytrain_POD = torch.from_numpy(Vytrain_POD.reshape(ntrain, ntimes, trunc_dim))
    Vyvalid_POD = torch.from_numpy(Vyvalid_POD.reshape(nvalid, ntimes, trunc_dim))
    Vytest_POD = torch.from_numpy(Vytest_POD.reshape(ntest, ntimes, trunc_dim))
    MUtrain = MUtrain.reshape(ntrain, ntimes, nparams)
    MUvalid = MUvalid.reshape(nvalid, ntimes, nparams)
    MUtest = MUtest.reshape(ntest, ntimes, nparams)

    for i in range(ntrain):
        for j in range(ntimes):
            train_loader_svd.store(Vxtrain_POD[i][j], Vytrain_POD[i][j], MUtrain[i][j], timesteps_vec[j])

    for i in range(nvalid):
        for j in range(ntimes):
            valid_loader_svd.store(Vxvalid_POD[i][j], Vyvalid_POD[i][j], MUvalid[i][j], timesteps_vec[j])

    for i in range(ntest):
        for j in range(ntimes):
            test_loader_svd.store(Vxtest_POD[i][j], Vytest_POD[i][j], MUtest[i][j], timesteps_vec[j])

    # POD RECONSTRUCTION ERRORS ON TEST DATA
    # Vytrain_reconstructed = torch.from_numpy(Vytrain_reconstructed.reshape(ntrain, ntimes, nvelocity // 2))
    # Vyvalid_reconstructed = torch.from_numpy(Vyvalid_reconstructed.reshape(nvalid, ntimes, nvelocity // 2))
    Vxtest_reconstructed = torch.from_numpy(Vxtest_reconstructed.reshape(ntest, ntimes, nvelocity // 2))
    Vytest_reconstructed = torch.from_numpy(Vytest_reconstructed.reshape(ntest, ntimes, nvelocity // 2))

    Vtest_reconstructed = torch.zeros(ntest, ntimes, nvelocity)
    Vtest_reconstructed[:, :, 0: nvelocity: 2] = Vxtest_reconstructed
    Vtest_reconstructed[:, :, 1: nvelocity: 2] = Vytest_reconstructed

    print("Mean relative POD reconstruction error on V: %s" % num2p(mre(Vtest, Vtest_reconstructed)))

    # Vtest = Vtest.reshape(ntest*ntimes, -1)
    # Vtest_reconstructed = Vtest_reconstructed.reshape(ntest*ntimes, -1)

    projection_error = mre(Vtest, Vtest_reconstructed)

    print("Computed SVD on the whole trainset with projection error (testset) due to truncation of:", projection_error)

    return Wx, Wy, train_loader_svd, test_loader_svd, valid_loader_svd, projection_error, scalerVx, scalerVy

class TimeSeriesDataset(torch.utils.data.Dataset):
    '''
    Input: sequence of input measurements with shape (ntrajectories, ntimes, ninput) and corresponding measurements of high-dimensional state with shape (ntrajectories, ntimes, noutput)
    Output: Torch dataset
    '''

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.len = X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.len


def Padding(data, lag):
    '''
    Extract time-series of lenght equal to lag from longer time series in data, whose dimension is (number of time series, sequence length, data shape)
    '''

    data_out = torch.zeros(data.shape[0] * data.shape[1], lag, data.shape[2])

    for i in range(data.shape[0]):
        for j in range(1, data.shape[1] + 1):
            if j < lag:
                data_out[i * data.shape[1] + j - 1, -j:] = data[i, :j]
            else:
                data_out[i * data.shape[1] + j - 1] = data[i, j - lag: j]

    return data_out


def multiplot(yts, plot, titles=None, fontsize=None, figsize=None, vertical=False, axis=False, save=False,
              name="multiplot"):
    """
    Multi plot of different snapshots
    Input: list of snapshots, related plot function, plot options, save option and save path
    """

    plt.figure(figsize=figsize)
    for i in range(len(yts)):
        if vertical:
            plt.subplot(len(yts), 1, i + 1)
        else:
            plt.subplot(1, len(yts), i + 1)
        plot(yts[i])
        if titles is not None:
            plt.title(titles[i], fontsize=fontsize)
        if not axis:
            plt.axis('off')

    if save:
        plt.savefig(name.replace(".png", "") + ".png", transparent=True, bbox_inches='tight')


def trajectory(yt, plot, title=None, fontsize=None, figsize=None, axis=False, save=False, name='gif'):
    """
    Trajectory gif
    Input: trajectory with dimension (sequence length, data shape), related plot function for a snapshot, plot options, save option and save path
    """

    arrays = []

    for i in range(yt.shape[0]):
        plt.figure(figsize=figsize)
        plot(yt[i])
        plt.title(title, fontsize=fontsize)
        if not axis:
            plt.axis('off')
        fig = plt.gcf()
        display(fig)
        if save:
            arrays.append(np.array(fig.canvas.renderer.buffer_rgba()))
        plt.close()
        clc(wait=True)

    if save:
        imageio.mimsave(name.replace(".gif", "") + ".gif", arrays)


def trajectories(yts, plot, titles=None, fontsize=None, figsize=None, vertical=False, axis=False, save=False,
                 name='gif'):
    """
    Gif of different trajectories
    Input: list of trajectories with dimensions (sequence length, data shape), plot function for a snapshot, plot options, save option and save path
    """

    arrays = []

    for i in range(yts[0].shape[0]):

        plt.figure(figsize=figsize)
        for j in range(len(yts)):
            if vertical:
                plt.subplot(len(yts), 1, j + 1)
            else:
                plt.subplot(1, len(yts), j + 1)
            plot(yts[j][i])
            plt.title(titles[j], fontsize=fontsize)
            if not axis:
                plt.axis('off')

        fig = plt.gcf()
        display(fig)
        if save:
            arrays.append(np.array(fig.canvas.renderer.buffer_rgba()))
        plt.close()
        clc(wait=True)

    if save:
        imageio.mimsave(name.replace(".gif", "") + ".gif", arrays)


def max_min(S, n_train):
    S_max = np.max(np.max(S[:n_train], axis = 1), axis = 0) # np.max(S, axis = 1) -> max of each row
    S_min = np.min(np.min(S[:n_train], axis = 1), axis = 0)

    return S_max, S_min

def scaling(S, S_max, S_min):
    S = (S - S_min)/(S_max - S_min)
    return S

def inverse_scaling(S, S_max, S_min):
    S = (S_max - S_min) * S + S_min
    return S

def normalize(data_loader, min=None, max=None):
    data = data_loader.get_all_samples()

    snapshots = data['snapshots']
    mus = data['mus']
    dts = data['dts']

    if min is None:
        min = np.min(snapshots)
    if max is None:
        max = np.max(snapshots)

    scaled_snapshots = scaling(snapshots, min, max)

    data_loader.size, data_loader.pts = 0, 0
    for i in range(scaled_snapshots.shape[0]):
        data_loader.store(scaled_snapshots[i], mus[i], dts[i])

    return min, max

def lipschitz_regularization_loss(model, snapshots, epsilon=0.1):
    v = torch.randn(snapshots.shape, device=snapshots.device)
    v = (v.view(v.shape[0], -1) / torch.norm(v.view(v.shape[0], -1), dim=0)).view(v.shape)  # normalize to unit vector
    latent_snapshots = model(snapshots)
    latent_snapshots_noise = model(snapshots + epsilon * v)
    J_forward_penalty = torch.norm(latent_snapshots - latent_snapshots_noise) ** 2 / epsilon
    return J_forward_penalty

def apply_xy_encode(xy, split_dim: int, funcx, funcy):
    assert xy.shape[split_dim] == 2
    if isinstance(xy, torch.Tensor):
        x_transformed = funcx(torch.index_select(xy, split_dim, torch.tensor([0])))
        y_transformed = funcy(torch.index_select(xy, split_dim, torch.tensor([1])))
        return torch.cat((x_transformed, y_transformed), dim=split_dim)
    x_transformed = funcx(xy.take(0, axis=split_dim))
    y_transformed = funcx(xy.take(1, axis=split_dim))
    return np.concatenate((x_transformed, y_transformed), axis=split_dim)

def apply_xy_decode(xy, split_dim: int, funcx, funcy):
    assert xy.shape[split_dim] % 2 == 0
    if isinstance(xy, torch.Tensor):
        x_transformed = funcx(torch.index_select(xy, split_dim, torch.arange(xy.shape[split_dim] // 2, device=xy.device)))
        y_transformed = funcy(torch.index_select(xy, split_dim, torch.arange(xy.shape[split_dim] // 2, xy.shape[split_dim], device=xy.device)))
        return torch.cat((x_transformed, y_transformed), dim=split_dim)
    raise NotImplementedError("Not implemented for numpy")
    return np.concatenate((x_transformed, y_transformed), axis=split_dim)

def transform(*args, mode='decode', xy_axis=1, scaler=None, svd=None, ae=None, device=None):
    transforms = []
    if mode == 'encode':
        raise NotImplementedError("Encoding transforms not implemented.")
    elif mode == 'decode':
        if ae is not None:
            transforms.append(lambda x: ae.decode(x))
        if scaler is not None:
            raise NotImplementedError("Scaling transforms not implemented.")
        if svd is not None:
            # TODO properly store svd
            if device is None:
                decode_x = lambda x: x @ svd[()]['svd_vx'][-1]
                decode_y = lambda y: y @ svd[()]['svd_vy'][-1]
            else:
                decode_x = lambda x: x @ torch.from_numpy(svd[()]['svd_vx'][-1]).to(device)
                decode_y = lambda y: y @ torch.from_numpy(svd[()]['svd_vy'][-1]).to(device)
            transforms.append(lambda xy: apply_xy_decode(xy, xy_axis, decode_x, decode_y))
    else:
        raise ValueError('mode must be encode or decode')

    transformed = []
    for arg in args:
        x = arg
        for t in transforms:
            x = t(x)
        transformed.append(x)
    return tuple(transformed)


def load_trajectories(filename):
    file, ext = os.path.splitext(filename)
    if ext == ".npz":
        data = np.load(filename)
        v = data['v']
        p = data['p']
        mu = data['mu']
    elif ext == ".hdf5":
        f = h5py.File(filename, "r")
        v = f["V"]
        p = f["P"]
        mu = f["MU"]
    else:
        NotImplementedError("File extension must be .npz or .hdf5")
    return v, p, mu


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)