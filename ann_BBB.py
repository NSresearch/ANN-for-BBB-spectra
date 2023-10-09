########################################################################
########################################################################
########################################################################
################ OBBB ANN training on CSV data##########################
########################################################################
########################################################################
########################################################################


import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras import initializers
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from tensorflow.keras import regularizers


import os

def ROC (ans, y_test, n = 100):
    
    #print(ans)
    
    roc = []
    for th in np.linspace(min(ans[:,0]) - 0.1*min(ans[:,0]), max(ans[:,0]) + 0.1*max(ans[:,0]), num=n):

        TP = 0
        TN = 0
        FP = 0
        FN = 0

        for i in range(len(ans)):
            if ans[i,0] > th and y_test[i,0] > th:
                TP += 1
                
            if ans[i,0] < th and y_test[i,0] < th:
                TN += 1
                
            if ans[i,0] > th and y_test[i,0] < th:
                FN += 1
                
            if ans[i,0] < th and y_test[i,0] > th:
                FP += 1
                
        if (TP + FN) != 0 and (FP+TN) != 0:
            TPR = TP/(TP + FN)
            FPR = FP/(FP + TN)
        else:
            if TP == 0:
                FPR = 0
                TPR = 0
            else:
                FPR = 1
                TPR = 1
        #print(th,"\t", FPR,"\t", TPR)
        roc.append([FPR,TPR])
        
    roc = np.array(roc)
    plt.title("ROC curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.plot([0, 1],[0, 1])
    
    plt.scatter(roc[:,0], roc[:,1], c=np.linspace(min(ans[:,0]), max(ans[:,0]), num=n))
    plt.plot(roc[:,0], roc[:,1])
    plt.show()
    return(roc)

LOAD =0


filelist = os.listdir(".")

print(filelist)

points = 5

#for ch in (".1", ""):
data = []
X = np.empty([1,points])
y = np.empty([1])
colors = ("blue", "orange", "green", "red", "purple", "brown", "pink", "gray")


filelist = os.listdir(".")

print(filelist)


for fname in filelist:
    if (".csv" in fname) and ("BBB" in fname):
        print(fname)
        df = pd.read_csv(fname)
        df = df[df["time"]>1800]
        df = df[df["time"]<3600]
        ch = ".1"
        #print(df)
        data = df[["std EEG1", "std EEG2", "0","1","2","3","4","5","0"+ch,"1"+ch,"2"+ch,"3"+ch,"4"+ch,"5"+ch]]        
        #data = df[["std EEG1", "std EEG2","2","3","4","5","2"+ch,"3"+ch,"4"+ch,"5"+ch]]
        #data = df[["std EEG1", "std EEG2","0","1","2","3","0"+ch,"1"+ch,"2"+ch,"3"+ch]]   
        
        data["BBB"] = [0 if  "before" in fname else 1]*len(data)
        

        X_l = np.array(data.iloc[:, 2:14].values)
        y_l = np.array(data.iloc[:, 14].values)
        X_l = X_l[:, 1:6] + X_l[:, 7:12]
        X_l =np.log(X_l)
        
        X = np.concatenate((X, X_l), axis = 0)
        y = np.concatenate((y, y_l), axis = 0)

y[0] = 0

print("X: ", len(X), X)
print("y: ", len(y), y)


            #print(i, j, X[i,j])
# X_t = dataset_test.iloc[:, 00:20].values
# y_t = dataset_test.iloc[:, 20:21].values

# print(X)
# print(X[::10])
# print(y)
# Normalizing the data

# sc = StandardScaler()
# X = sc.fit_transform(X)
# print('Normalized data:')
# print(X[0])

# Нормируем на минимум и максимум
#
#for i in range(len(X)):
    #X[i,2:points] = np.log(X[i,2:points])
    #A_min = np.min(X[i,2:points])
    #A_max = np.max(X[i,2:points])
    #print(A_min, A_max)

    #X[i,2:points] = (X[i,2:points] - A_min) / (A_max - A_min)
    #print(X[0])
    
#A_min = np.min(X[:,0:2])
#A_max = np.max(X[:,0:2])

#print("STD min, max: ", A_min, A_max)

#X[:,0:2] = (X[:,0:2] - A_min) / (A_max - A_min)


for i in range(len(X)):
                    A_min = np.min(X[i,:])
                    A_max = np.max(X[i,:])
                    #print(A_min, A_max)

                    #X[i,:] = (X[i,:] - A_min) / (A_max - A_min)
                    X[i,:] = X[i,:]/sum(X[i,:])

labels = ('gamma','beta', 'alpha', 'theta', 'delta', 'low')                    
for i in range(5):
    plt.subplot(5,1,i+1)
    plt.scatter(range(len(X)), X[:,i], s = y+1, c = y)

    plt.colorbar()
    plt.title(labels[i])
    plt.ylim((-0.1,.6))
    plt.grid()
plt.show()

                    
for i in range(len(X)):
    for j in range(points):
        if np.isnan(X[i,j]):
            X[i,j] = 0
                    

    
for i in range(len(X)):
    for j in range(points):
        if np.isnan(X[i,j]):
            X[i,j] = 0
print("X: ", len(X), X)
print("y: ", len(y), y)

# One hot encode

ohe = OneHotEncoder()
y = ohe.fit_transform(y.reshape(-1, 1)).toarray()
print('One hot encoded array:')
print("y after OHE: ", y)
# print(y_t[0:15])

# Train test split of model

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)
# print("X_train: ", X_train)
# print("y_train: ", y_train)
# print("X_test: ", X_test)
# print("y_test: ", y_test)
# X_train, y_train = X, y
# X_test, y_test = X_t, y_t

if LOAD == 0:

    fac = 'selu'

    model = Sequential()
    model.add(Dense(points, input_dim=points, activation=fac,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros", 
        kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5)))


    model.add(Dense(150, activation=fac,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros", 
        kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5)))


    model.add(Dense(2, activation='softmax'))#, kernel_initializer=initializers.Ones(),    bias_initializer=initializers.Zeros()))



    # To visualize neural network
    model.summary()

    model.compile(loss='CategoricalCrossentropy', optimizer='adamax', metrics=['accuracy'])

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=2048)


    model.save("Glioma")


    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    
else:
    model = load_model("Glioma")

np.savetxt("roc.dat", ROC( y_test, model.predict(X_test)), fmt='%1.4e')

ans = np.array(model.predict(X_test))
for i in range(len(ans)):
    if ans[i,0] > 0.5:
        ans[i] = [1, 0]
    else:
        ans[i] = [0, 1]
        
#print(y_test, ans)
print(classification_report( y_test, ans))

