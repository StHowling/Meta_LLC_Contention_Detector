import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
# import os

default_cpu_range=[(0.95,1.01)]

def load_data(filename):
    '''
    expect the file to be a .csv or .npy, Nx5 array, where the last colomn is label 
    '''
    if filename[-4:] == '.csv':
        df = pd.read_csv(filename)
        data = df.values
    elif filename[-4:] == '.npy':
        data = np.load(filename)

    if data.shape[1] != 5:
        print("Error: expected dimension ( , 5), found dimension (%s, %s)" % (data.shape[0],data.shape[1]))
        exit()

    X = data[:,:4]
    Y = data[:,-1]
    return X,Y

def train_test_split(data,ratio=0.5):
    '''
    Since the data is essentially a time series and we intend to use history to infer the future,
    there is no meaning in shuffling and we just split between history and future
    '''
    training_lines=int(data.shape[0]*ratio)
    training_set=data[:training_lines]
    testing_set=data[training_lines:]

    return training_set,testing_set

def groupby_cpu(X,range_list=default_cpu_range):
    X_, X_train = {}, {}
    for r in range_list:
        X_[r] = []
    
    for i in range(X.shape[0]):
        flag = True
        for r in range_list:
            if X[i][-1] >= r[0] and X[i][-1] < r[1]:
                X_[r].append(X[i])
                flag = False
                break
        if flag:
            print("Warning: CPU usage at Line %s (%s) does not have its corresponding range in provided range list. We will skip this data point, better check the data distribution and renew the range list." % (i,X[i][-1]))
        
    for r in range_list:
        X_train[r] = np.array(X_[r])

    return X_train

def visualize_data_distribution_1d(x):
    '''
    Visualize density of a specific dimension
    '''
    l = np.min(x)
    u = np.max(x)
    interval = (u-l) / 20
    x_pos = np.linspace(l-interval,u+interval)
    y_cnt = np.zeros_like(x_pos)

    for item in x:
        for i in range(x_pos.shape[0]-1):
            if x_pos[i] <= item and x_pos[i+1] > item:
                y_cnt[i]+=1
                break

    y_cnt = y_cnt / x_pos.shape[0]

    plt.bar(x_pos,y_cnt)
    plt.show()

def visualize_data_distribution_2d(x,y):
    '''
    Visualize joint distribution of two dimensions
    '''
    plt.scatter(x,y,s=0.5)
    plt.show()

if __name__=="__main__":
    data_dir = "./data/"
    # data_file = sys.argv[1]
    data_file = "memcached.csv"

    X, Y = load_data(data_dir+data_file)
    X_train, X_test = train_test_split(X)
    
    visualize_data_distribution_1d(X_train[:,1]) # MPKI
    visualize_data_distribution_2d(X_train[:,1],X_train[:,2]) # MPKI, OCC