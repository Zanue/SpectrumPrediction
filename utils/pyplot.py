import numpy as np
import torch
import matplotlib.pyplot as plt
import math


def plot_seq_feature(pred_, true_, origin_, threshold_=None, label = "train"):
    assert(pred_.shape == true_.shape)

    pred = pred_.detach().clone()[..., -1:]
    true = true_.detach().clone()[..., -1:]
    origin = origin_.detach().clone()[..., -1:] 
    threshold = threshold_.detach().clone()[..., -1:] if threshold_ is not None else None 

    if len(pred.shape) == 3:  #BLD
        pred = pred[0]
        true = true[0]
        origin = origin[0]
        threshold = threshold[0] if threshold is not None else None
    pred = pred.cpu().numpy()
    true = true.cpu().numpy()
    origin = origin.cpu().numpy()
    threshold = threshold.cpu().numpy() if threshold is not None else None

    pred = np.concatenate([origin, pred], axis=0)
    true = np.concatenate([origin, true], axis=0)

    L, D = pred.shape
    # if D == 1:
    #     pic_row, pic_col = 1, 1
    # else:
    #     pic_col = 2
    #     pic_row = math.ceil(D/pic_col)
    pic_row, pic_col = D, 1


    fig = plt.figure(figsize=(8*pic_row,8*pic_col))
    for i in range(D):
        ax = plt.subplot(pic_row,pic_col,i+1)
        ax.plot(np.arange(L), pred[:, i], label = "pred")
        ax.plot(np.arange(L), true[:, i], label = "true")
        if threshold is not None:
            ax.plot(np.arange(origin.shape[0], L), threshold[:, i], label = "threshold")
        ax.set_title("dimension = {},  ".format(i) + label)
        ax.legend()

    return fig


def plot_heatmap_feature(pred_, true_, label = "pred"):
    assert(pred_.shape == true_.shape)
    pred = pred_
    true = true_

    if len(pred.shape) == 3:  #BLD
        pred = pred[0]
        true = true[0]

    pred = pred.T # [D, L]
    true = true.T # [D, L]

    fig = plt.figure(figsize=(16, 8)) 
    ax = plt.subplot(1,3,1)
    plt.imshow(pred)
    plt.colorbar()
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    ax.set_title("pred")

    ax = plt.subplot(1,3,2)
    plt.imshow(true)
    plt.colorbar()
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    ax.set_title("true")

    ax = plt.subplot(1,3,3)
    plt.imshow((pred-true))
    plt.colorbar()
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    ax.set_title("error bar")

    return fig

