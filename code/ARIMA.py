import numpy as np
import pandas as pd
from numpy.linalg import solve

class LeastSquares:
    def fit(self,X,y):
        self.w = solve(X.T@X, X.T@y)

    def predict(self, X):
        return X@self.w

class LassoRegression() : 
    # ref. https://www.geeksforgeeks.org/implementation-of-lasso-regression-from-scratch-using-python/
    def __init__(self, learning_rate, iterations, l1_penality): 
        self.learning_rate = learning_rate 
        self.iterations = iterations 
        self.l1_penality = l1_penality 
          
    def fit(self, X, Y) : 
        # no_of_training_examples, no_of_features 
        self.m, self.n = X.shape 
        # weight initialization 
        self.W = np.zeros(self.n) 
        self.b = 0
        self.X = X 
        self.Y = Y 
        # gradient descent learning   
        for i in range(self.iterations): 
            self.update_weights() 
        print(self.W)
        print(self.b)
        return self
      
    # Helper function to update weights in gradient descent 
    def update_weights(self):   
        Y_pred = self.predict(self.X) 
        # calculate gradients   
        dW = np.zeros(self.n) 
        for j in range(self.n): 
            if self.W[j] > 0 : 
                dW[j] = (-(2 * (self.X[:, j]).dot(self.Y - Y_pred)) + self.l1_penality) / self.m 
            else : 
                dW[j] = (-(2 * (self.X[:, j]).dot(self.Y - Y_pred)) - self.l1_penality) / self.m 
  
        db = - 2 * np.sum(self.Y - Y_pred) / self.m  
        # update weights 
        self.W = self.W - self.learning_rate * dW 
        self.b = self.b - self.learning_rate * db 
        return self

    def predict( self, X ) : 
        return X.dot(self.W) + self.b 

class ARIMA():
    def __init__(self, learning_rate, iterations, l1_penality, lag):
        self.learning_rate = learning_rate 
        self.iterations = iterations 
        self.l1_penality = l1_penality
        self.lag = lag

    def create_X(self, df, last_row, col_index):
        X = np.ones((1, self.lag))
        
        for i in range(10-self.lag):
            cur_time_step = df.iloc[last_row-self.lag-i : last_row-i, col_index].to_numpy().flatten().reshape(1, self.lag)
            X = np.vstack([X, cur_time_step])
        
        # Necessary operation: delete first initialization row (all 1's), add a column of 1's (intercept), flip
        X = np.delete(X, 0, axis = 0)
        X = np.c_[np.ones(10-self.lag), X]
        X = np.flip(X, axis=0)
        return X

    def create_y(self, df, last_row, col_index):
        return df.iloc[last_row-10+self.lag:last_row, col_index]

    def AR(self, df):
        pred_vec = np.ones((1,22))

        # transform X
        for row in range(11,41):
            for col in range(2):
                X = self.create_X(df, last_row=row-1, col_index=col)
                y = self.create_y(df, last_row=row, col_index=col)
                model = LeastSquares()
                model.fit(X,y)
                X_test = self.create_X(df, last_row=row, col_index=col)
                pred_y = model.predict(X_test)
                pred_vec[0,col] = pred_y[-1]
        
            df_pred = pd.DataFrame(data=pred_vec, columns=df.columns)
            df = df.append(df_pred, ignore_index=True)

        return df.iloc[-30:,:]

    def MA(self, df_X, df_y):
        true_y_bar = np.median(df_y.to_numpy(),axis=0)
        pred_y_bar = np.median(df_X.iloc[:,:2].to_numpy(),axis=0)
        return true_y_bar - pred_y_bar

    def fit(self, df_X, df_y):
        # staionarize
        df_X.iloc[:,2:] = df_X.iloc[:,2:].replace(0, np.nan)
        df_X = pd.DataFrame(df_X.diff())
        df_X = pd.DataFrame(np.exp(df_X))
        AR_ret = self.AR(df=df_X).reset_index(drop=True)
        df_y_first_row = df_y.iloc[0,:]
        df_y_first_row = np.exp(df_y_first_row)
        df_y = pd.DataFrame(df_y.diff())
        df_y = pd.DataFrame(np.exp(df_y))
        df_y.iloc[0,0] = df_y_first_row[0]
        df_y.iloc[0,1] = df_y_first_row[1]
        self.shift_factor_y = self.MA(AR_ret, df_y)
    
    def predict_pre_process(self, df_X_test):
        df_X_test.iloc[:,2:] = df_X_test.iloc[:,2:].replace(0, np.nan)
        df_X_test = pd.DataFrame(df_X_test.diff())
        df_X_test = pd.DataFrame(np.exp(df_X_test))
        return self.AR(df=df_X_test).reset_index(drop=True)

    def predict(self, AR_ret):
        pred_y = AR_ret.iloc[:,:2]
        pred_y_np = pred_y + self.shift_factor_y
        pred_y_np[pred_y_np < 0] = 0.37
        pred_y_np[pred_y_np > 2.7] = 2.7
        pred_y_np = np.log(pred_y_np)
        pred_y_first_row = np.zeros((1,2))
        pred_y_np = np.vstack((pred_y_first_row, pred_y_np)).cumsum(axis=0)
        pred_y_np = np.delete(pred_y_np, 0, axis = 0)
        return pd.DataFrame(data=pred_y_np, columns=pred_y.columns)