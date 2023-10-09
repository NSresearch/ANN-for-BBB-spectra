########################################################################
########################################################################
########################################################################
################ OBBB ANN testing on "Rat*.*.csv"#######################
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

from tensorflow.keras import regularizers


import os

LOAD = 1


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

model = load_model("Glioma") #_good0")

filelist = os.listdir("./")

points = 5
th=0.25
print(filelist)


#for ch in (".1", ""):
data = []
X = pd.DataFrame()
y = []

X_t = pd.DataFrame()
y_t = []
colors = ("blue", "orange", "green", "red", "purple", "brown", "pink", "gray")

POS = np.zeros((2,8,5))
NEG = np.zeros((2,8,5))
QUAN = np.zeros((2,8,5))

fig, axes = plt.subplots(nrows=7, ncols=2, figsize=(7, 7))
fig.tight_layout()

Normal_state = []
Anest_state = []
Lethal_anest = []

Normal_state_isof = []
Anest_state_isof = []
Lethal_anest_isof = []

for fname in  filelist:
            if ".csv" in fname and fname[0]!= "." and "BBB" not in fname:
                df = pd.read_csv(fname)
                ch = ".1"
                df = df[["std EEG1", "std EEG2", "0","1","2","3","4","5","0"+ch,"1"+ch,"2"+ch,"3"+ch,"4"+ch,"5"+ch]]
                
                X = np.array(df.iloc[:, 2:14].values)
                X = X[:, 1:6] + X[:, 7:12]
                X = np.log(X)
                
                norm_coeffs = np.sum(X, axis = 1)
                
                for i in range(len(X)):
                    A_min = np.min(X[i,:])
                    A_max = np.max(X[i,:])
                    #print(A_min, A_max)

                    #X[i,:] = (X[i,:] - A_min) / (A_max - A_min)
                    X[i,:] = X[i,:]/sum(X[i,:])
                for i in range(len(X)):
                    for j in range(points):
                        if np.isnan(X[i,j]):
                            X[i,j] = 0
                            
                y_a = model.predict(X)
                t_a = np.linspace(0, len(y_a)*10/60, len(y_a))
                plt.plot(t_a, y_a[:,1], alpha = 0.75)
                plt.show()
                
                y_a_sl_win = []
                win = 100
                for i in range(len(y_a)):
                    y_a_sl_win.append(sum(y_a[i-win//2:i+win//2, 1])/win)
                    
                y_a_th = np.zeros(len(y_a[:,1]))
                
                
                th = np.mean(y_a_sl_win[win//2:len(y_a_sl_win)-win//2])#0.75
                
                
                for i in range(len(y_a_th)):
                    if y_a_sl_win[i] > th:
                        y_a_th[i] = 1
                    else:
                        y_a_th[i] = 0
                    
                plt.plot(np.linspace(0, len(y_a)*10/60, len(y_a_sl_win)), y_a_sl_win)
                plt.plot(t_a, y_a_th, alpha = 0.5)
                plt.plot((0,200), (th,th), alpha = 0.25)
                
                if "1." in fname:
                    Normal_state.extend(y_a_sl_win[6*0:6*25])
                    Anest_state.extend(y_a_sl_win[6*35:6*145])
                    Lethal_anest.extend(y_a_sl_win[6*155:])
                    
                    
                else:
                    
                    Normal_state_isof.extend(y_a_sl_win[6*0:6*25])
                    Anest_state_isof.extend(y_a_sl_win[6*35:6*55])
                    Lethal_anest_isof.extend(y_a_sl_win[6*65:6*110])
                    
                
                a = np.asarray([t_a, y_a[:,1], y_a_sl_win, y_a_th, np.ones(len(t_a))*th ]).T
                print(a)
                np.savetxt(fname[:-3]+"dat", a, delimiter ='\t')
                
                
                plt.xlabel("t, min")
                plt.ylabel("BBB state")
                plt.ylim((-0.1,1.1))
                plt.xlim((0,200))
                plt.title(fname)
                
plt.show()

#Normal_state = np.array(Normal_state).flatten().flatten()

print(Normal_state, Anest_state, Lethal_anest)

xaxis = ((0,1, 3,4, 6,7))

plt.boxplot((Normal_state, Normal_state_isof, Anest_state, Anest_state_isof, Lethal_anest, Lethal_anest_isof), showfliers=False)
plt.scatter((1,2,3,4,5,6), (np.mean(Normal_state), np.mean(Normal_state_isof), np.mean(Anest_state), np.mean(Anest_state_isof), np.mean(Lethal_anest), np.mean(Lethal_anest_isof)))
plt.show()

