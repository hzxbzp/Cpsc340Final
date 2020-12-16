import numpy as np
import pandas as pd
import ARIMA
import time

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
weights_list = [None] * 20

for number in range(20):
    test_path = '../data/test/X/X_{}.csv'.format(number)
    data = np.genfromtxt(test_path, delimiter=",")
    data = data[data[:, 0] == 0]
    idx = np.arange(10, 61, 6)
    idx = np.append(idx, np.arange(11, 61, 6))
    idx = np.sort(idx)
    data = data[:, [idx]]
    data = np.reshape(data, (9, 2))
    weights = comp_data(csv, data)
    weights = np.delete(weights,593,axis=0)
    weights = np.delete(weights,858,axis=0)
    weights = np.delete(weights,2166,axis=0)
    weights_list[number] = weights

################################################################################

df_train_X = pd.read_csv('../data/train_transformed.csv')
df_train_y = pd.read_csv('../data/ytrain_transformed.csv')

# X_training dim 11 (-1000 to 0) * 22 (agent_x to rel_y9)
X_list = [pd.DataFrame()] * 2308
for i in range(2308):
    X_list[i] = df_train_X.iloc[(i*11):(i*11+11), 1:]

# y_training dim 30 * 2 (the first is x and the second is y)
y_list = [pd.DataFrame()] * 2308
for i in range(2308):
    y_list[i] = df_train_y.iloc[(i*30):(i*30+30),-2:]

# preprocessing
del X_list[593]
del y_list[593]
del X_list[858]
del y_list[858]
del X_list[2166]
del y_list[2166]

# 2308 total samples, 2305 good samples
model_list = [ARIMA.ARIMA(learning_rate=1, iterations=1000, l1_penality=1, lag=3)] * 2308
t = time.time()
num_test = 2305
for i in range(num_test):
    model_list[i].fit(X_list[i], y_list[i])
    print(str(i) + " is okay.")
print("Fitting took %d seconds" % (time.time()-t))

df_test_X = pd.read_csv('../data/test_transformed.csv')
X_test_list = [pd.DataFrame()] * 20
for i in range(20):
    X_test_list[i] = df_test_X.iloc[(i*11):(i*11+11), 1:]

df_Y = df_train_X.iloc[0:1,1:3]
for i_test in range(20):
    pred_y = np.zeros((30,2))
    for j_model in range(num_test):
        pred_y += weights_list[i][j_model, 0] * model_list[j_model].predict(X_test_list[i_test])
    df_Y = df_Y.append(pred_y, ignore_index=True)

df_Y = df_Y.iloc[1:,:].reset_index(drop=True)
df_Y = df_Y.stack()
df_Y.to_csv("output.csv", index=False)

################################################################################