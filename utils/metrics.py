import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    sigma_p = pred.std(axis=0)
    sigma_g = true.std(axis=0)
    mean_p = pred.mean(axis=0)
    mean_g = true.mean(axis=0)
    index = (sigma_g != 0)
    corr = ((pred - mean_p) * (true - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    return (corr[index]).mean()

def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    rse = RSE(pred, true)
    corr = CORR(pred, true)
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return rse, corr, mae, mse, rmse, mape, mspe
