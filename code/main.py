import numpy as np
import pandas as pd
import ARIMA

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

################################################################################

df = pd.read_csv('../data/train_transformed.csv')
# X_training dim 11 (-1000 to 0) * 22 (agent_x to rel_y9)
df_train_1_X = df.iloc[0:11, 1:]

model = ARIMA.ARIMA(learning_rate=1, iterations=1000, l1_penality=1, lag=3)

df2 = pd.read_csv('../data/train/y/y_0.csv')
df_train_1_y = df2.iloc[:,-2:]
# y_training dim 30 * 2 (the first is x and the second is y)
model.fit(df_train_1_X, df_train_1_y)

# X_test dim 11 * 22 same as X_training
df_train_2_X = df.iloc[11:22, 1:]

# return 30 * 2 dataframe
modelret = model.predict(df_train_2_X)
print(modelret)

################################################################################