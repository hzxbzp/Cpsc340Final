import numpy as np


# Referenced https://datascience.stackexchange.com/questions/33081/similarity-between-two-scatter-plots
def comp_data(models, data):
    w = np.zeros((len(models), 1))
    dcenter = np.mean(data, axis=0)
    dstd = np.std(data)
    for i in range(len(models)):
        mcenter = np.mean(models[i], axis=0)
        mstd = np.std(models[i])
        cdist = np.linalg.norm(dcenter - mcenter)
        sdist = np.abs(dstd - mstd)
        w[i] = 1 / (cdist + sdist)

    return w


csv = np.genfromtxt('../data/train_transformed.csv', delimiter=",")
csv = csv[csv[:, 0] == 0]
csv = np.delete(csv, [0, 1, 2], 1)
csv = np.reshape(csv, (csv.shape[0], 10, 2))

data = np.genfromtxt('../data/test/X/X_0.csv', delimiter=",")
data = data[data[:, 0] == 0]
idx = np.arange(10, 61, 6)
idx = np.append(idx, np.arange(11, 61, 6))
idx = np.sort(idx)
data = data[:, [idx]]
data = np.reshape(data, (9, 2))

weights = comp_data(csv, data)