########################################################################
########################################################################
########################################################################
############### EDF reading and FFT-based features #####################
################ calculation with export to *.csv ######################
########################################################################
########################################################################
########################################################################



import pandas as pd
import numpy as np
import pyedflib
import matplotlib.pyplot as plt
from scipy.fft import fft

import os

def edf_to_dataframe(import_edf_file, channels, averaging_window_sec=60, window_shift_sec=10, export_csv_file=None) :
    df=None
    f = None    
    f = pyedflib.EdfReader(import_edf_file)
    signal_labels = f.getSignalLabels()
    print(signal_labels)
    print("EEG channels: ",channels)
    
    gamma = (30,50)
    beta = (12,30)
    alpha = (8,12)
    theta = (4,8)
    delta = (1,4)
    low = (0,1)
    
    bands = (gamma,beta,alpha,theta, delta,low)

    
    n = f.signals_in_file
    T0=f.getStartdatetime()
    freq = int(f.getSampleFrequency(1))
    print("Recording start: ",0)
    print("Frequency = ",freq," Hz")

    if (len(channels)+2 != n) | (sum(f.getNSamples()[channels])/len(channels)!=f.getNSamples()[channels[0]]) :
        print("ERROR: Возможно не соответствие количества каналов!")
    else :
        num_points = f.getNSamples()[channels[0]]
        averaging_window_points = averaging_window_sec*freq
        window_shift_points = window_shift_sec*freq
        num_averaging = int((num_points-averaging_window_points)/window_shift_points)+1
        print(num_points," points in one realization or ",num_points/freq," seconds.")

        DATA = {}
        for j in range(len(channels)) :
            EEG = f.readSignal(channels[j-1])
            mean_values = []
            std_values = []
            time_values_sec = []
            band_powers = []
            for i in range(num_averaging) :
                t_sec = i*window_shift_sec+averaging_window_sec
                t_points = t_sec*freq
                averaging_data = EEG[t_points-averaging_window_points : t_points]
                mean = np.mean(averaging_data)
                std = np.std(averaging_data)
                mean_values.append(mean)
                std_values.append(std)
                time_values_sec.append(t_sec)
                fft_values =fft(averaging_data).real[1:averaging_window_sec*freq//2] + fft(averaging_data).real[averaging_window_sec*freq:averaging_window_sec*freq//2:-1]
                fft_values = fft_values[0:averaging_window_sec*freq//40]
                fft_values = fft_values*fft_values
                #fft_values = fft_values/np.max(fft_values)
                one_point = freq/(len(averaging_data)/len(fft_values))/len(fft_values)

                
                powers = []
                for band in bands:
                    s = fft_values[int(band[0]/one_point):int(band[1]/one_point)].sum()/(len(fft_values[int(band[0]/one_point):int(band[1]/one_point)])*averaging_window_points)
                    powers.append(s)
                    #print(s)
                band_powers.append(powers/np.sum(powers))
                
            if j==0 :
                DATA["time"] = time_values_sec
                df_bands = pd.DataFrame(band_powers)
            else:
                df_bands = pd.concat([df_bands,pd.DataFrame(band_powers)], axis = 1)
            DATA["mean EEG"+str(j+1)] = mean_values
            DATA["std EEG"+str(j+1)] = std_values
            #DATA["Powers"+str(j+1)] = np.array(band_powers)
                

            plt.figure(figsize=(15,5))
            plt.subplot(2,2,j+1)
            plt.plot(time_values_sec,mean_values)
            plt.xlabel("time, sec")
            plt.ylabel("mean EEG "+str(j+1))

            plt.subplot(2,2,j+2)
            plt.plot(time_values_sec,std_values)
            plt.xlabel("time, sec")
            plt.ylabel("std EEG "+str(j+1))

		

        df = pd.concat([pd.DataFrame.from_dict(DATA), df_bands], axis=1)
        print(df)
        df.head()
        
        if export_csv_file :
            df.to_csv(export_csv_file)

    f.close()
    return df

filelist = os.listdir("./")

print(filelist)

for fname in filelist:
    if ".edf" in fname:

        df = edf_to_dataframe(import_edf_file=fname, channels=[1,2], 
                      averaging_window_sec=60, window_shift_sec=10, 
                      export_csv_file=fname[:len(fname) - 3]+"csv")
