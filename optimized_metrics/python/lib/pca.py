# Copyright (c) Baptiste Caramiaux, Etienne Thoret
# All rights reserved

import numpy as np
import tensorflow as tf
from tensorly.decomposition import robust_pca
import tensorly.backend as T

def adhoc_pca(data, num_comp):
    n, p = data.shape[0], data.shape[1]
    # subtract off the mean for each dimension
    mn = np.mean(data, axis=0)
    data = np.subtract(data, mn)
    # construct the matrix Y
    Y = data / np.sqrt(n - 1)
    # SVD does it all
    u, S, pc = np.linalg.svd(Y, full_matrices=False)
    # project the original data
    points = np.transpose(np.dot(pc, np.transpose(data)))
    # calculate the variances
    # variances = np.multiply(S, S)
    # find minimum nb of components needed to have var > threshold
    # cum_explained = np.cumsum(variances / np.sum(variances))
    # idx = np.where(cum_explained > threshold)[0]
    # return projected data
    # return points[:, :idx[0] + 1]
    return points[:, :num_comp], pc, S


# def foldn(An, shape, dim):
#     index = [shape[dim:], shape[:dim]]
#     C = np.reshape(An, index)
#     C = shiftdim(C,1+len(shape)-dim)
#     return C

# def tmuln(A, U, n):
#     if U.shape[1] != A.shape[n-1]
#         print('Inner matrix and tensor dimensions must agree')
#     index = list(A.shape)
#     C = foldn(U*unfoldn(A,n),[index(1:n-1) size(U,1) index(n+1:end)],n)


def pca_patil(tensor, nb_freq, n_components=1):
    # center and div by std
    # tensor = np.mean(tensor, axis=0)
    pcf=21
    pcr=4
    pcs=5
    pc_ = [pcf, pcs, pcr]

    Ut, St = [], []
    for t in range(tensor.shape[0]):
        U, S = [], []
        tensor_t = tensor[t,:,:,:]
        tensor_t = (tensor_t - np.mean(tensor_t)) / np.std(tensor_t)
        for i in range(len(tensor_t.shape)):
            if i != len(tensor_t.shape)-1:
                tens_shift = np.moveaxis(tensor_t, 0, -i)
            else:
                tens_shift = np.transpose(tensor_t)
            tens_shift = tens_shift.reshape((tens_shift.shape[0], tens_shift.shape[1] * tens_shift.shape[2]))
            Ui, Si, _ = np.linalg.svd( np.dot(tens_shift, np.transpose(tens_shift)))
            # print(tensor_t.shape, tens_shift.shape, Ui.shape)
            U.append(Ui)
            S.append(Si)
        Ut.append(U)
        St.append(S)

    Ut_map = []
    new_tensor = np.zeros((tensor.shape[0], pcs, pcr, pcf))
    for t in range(tensor.shape[0]):
        tensor_t = tensor[t,:,:,:]
        tensor_t = (tensor_t - np.mean(tensor_t)) / np.std(tensor_t)
        # index=size(A);
        # C=foldn(U*unfoldn(A,n),[index(1:n-1) size(U,1) index(n+1:end)],n);
        A = tensor_t
        Ut_map_i = []
        for i in range(len(tensor_t.shape)):
            U_pcs = np.transpose(Ut[t][i][:,:pc_[i]])
            Ut_map_i.append(U_pcs)
            # if i != len(tensor_t.shape)-1:
            tens_shift = np.moveaxis(A, 0, -i)
            # else:
            #     tens_shift = np.transpose(A)
            # print(A.shape, U_pcs.shape, tens_shift.shape)
            # print('A before:', A.shape, U_pcs.shape, tens_shift.shape)
            tens_shift = np.moveaxis(tens_shift, 0, 1)
            A = np.dot(U_pcs, tens_shift)
            # tens_shift = np.moveaxis(tens_shift, 0, -1)
            # print('A after:', A.shape)
        new_tensor[t,:,:,:] = np.moveaxis(A, 0, 1)
        Ut_map.append(Ut_map_i)
    ppcomps = Ut_map
    variances = []
    return new_tensor, ppcomps, variances



            
    # A = tensor
    # index = list(A.shape)
    # # A=tmuln(A,U{1}(:,1:pcs)',1);
    # U1 = U[0]
    # n = 1
    # unfoldn_A_n = np.moveaxis(A, 0, -n)
    # A=foldn(U1 * unfoldn_A_n, [index[0:n], U.shape[0], index[n:end]], n)
    # A=tmuln(A,U{2}(:,1:pcr)',2);
    # if length(size(A))<3 A=reshape(A,[1 size(A,1) size(A,2)]);end
    # A=tmuln(A,U{3}(:,1:pcf)',3);
    # A=(A-mean2(A))/std2(A);
    # print(np.mean(tensor))


def pca(tensor, nb_freq, n_components=1):
    # print('pca', tensor.shape, nb_freq)
    num_comp = n_components
    tensor_avg = np.mean(tensor, axis=0)
    # u,s,v = tf.linalg.svd(tensor)
    # print(u.shape, s.shape, v.shape)
    # with tf.Session():
    #     s = tf.linalg.svd(tensor_avg, compute_uv=False)
    #     print(s.shape)
    
    # print(tensor.shape, nb_freq, num_comp, tensor.shape[2])
    tensor_red = np.zeros((nb_freq, num_comp*tensor.shape[2]))
    ppcomps = []
    variances = []
    for freq_i in range(nb_freq):
        pts, pcs, varis = adhoc_pca(tensor_avg[freq_i, :, :], num_comp)
        tensor_red[freq_i, :] = np.transpose(pts.flatten())
        ppcomps.append(pcs)
        variances.append(varis)
    return tensor_red, ppcomps, variances