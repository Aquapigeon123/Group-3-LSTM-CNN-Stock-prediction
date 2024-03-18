import pandas as pd
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

#importation of data
filename = 'AAPL'
stock = pd.read_csv('../Data/AAPL.csv')


scaler = preprocessing.MinMaxScaler()
scaled_values = scaler.fit_transform(stock.iloc[:,1:4])
stock.iloc[:,1:4] = scaled_values

y_scaler = preprocessing.MinMaxScaler()
scaled_values = y_scaler.fit_transform(np.array(stock.iloc[:,4]).reshape(-1,1))
stock.iloc[:,4] = scaled_values

scaler = preprocessing.MinMaxScaler()
scaled_values = scaler.fit_transform(stock.iloc[:,5:])
stock.iloc[:,5:] = scaled_values



#data processing
window_size = 3
X = []
Y = []

for i in range(0 , len(stock) - window_size -1 , 1):
    X.append(np.array(stock.iloc[i:i+window_size,4]).reshape(window_size,1))
    Y.append(np.array(stock.iloc[i+window_size,4]).reshape(1,1))
train_X,test_X,train_label,test_label = train_test_split(X, Y, test_size=0.1,shuffle=False)
train_X = np.array(train_X)
test_X = np.array(test_X)
train_label = np.array(train_label)
test_label = np.array(test_label)

train_X = np.squeeze(train_X)
test_X = np.squeeze(test_X)
train_label = np.squeeze(train_label)


#find the best parameters
epsilon = 0.05
param_grid = {'C': [ 1, 10, 100],'gamma': [0.1, 1, 10,100],'epsilon':[0.01]}

# Creation of the model
svr = SVR(kernel='rbf')

# find best parameters
grid_search = GridSearchCV(svr, param_grid, cv=5)
grid_search.fit(train_X, train_label)

# print best parameters
print("Meilleurs paramètres trouvés : ", grid_search.best_params_)


    #training of the model
svr_rbf = grid_search.best_estimator_
svr_rbf.fit(train_X, train_label)

    #test
y_pred=svr_rbf.predict(test_X)
y_pred_train = svr_rbf.predict(train_X)
    
    #loss
n = len(y_pred)
loss=0
for i in range(n):
    loss += (y_pred[i] - test_label[i,0])**2
test_error=loss/n
    
N = len(y_pred_train)
loss=0
for i in range(N):
    loss += (y_pred_train[i] - train_label[i])**2
train_error =loss/n
    

test_label[:,0] = y_scaler.inverse_transform(test_label[:,0])
y_pred = np.array(y_pred).reshape(-1,1)
y_pred = y_scaler.inverse_transform(y_pred)


    #plot
plt.plot(test_label[:,0], color='darkorange', label='Stock price')
plt.plot(y_pred, color='navy', lw=2, label='Predicted stock price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('SVR with RBF kernel')
plt.legend()
plt.show()

print(train_error)
print(test_error)