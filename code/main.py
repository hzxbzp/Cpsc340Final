import numpy as np


#Referenced https://datascience.stackexchange.com/questions/33081/similarity-between-two-scatter-plots
def comp_data(models, data):
    w = np.zeros((len(models), 1))
    dcenter = np.mean(data, axis=0)
    dstd = np.std(data)
    for i in range(len(models)):
        mcenter = np.mean(models[i], axis=0)
        mstd = np.std(models[i])
        cdist = np.linalg.norm(dcenter-mcenter)
        sdist = np.abs(dstd-mstd)
        w[i] = 1/(cdist + sdist)

    return w
