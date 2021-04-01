import numpy as np
import pandas as pd
from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans

def ATNT100(file):
    if file == 'ATNTFaceImages400.txt':
        data = pd.read_csv(file, header=None, sep=",")
        dataframe = data.iloc[:, :100]
        datawithoutlabel = dataframe.iloc[1:, :]
        label = dataframe.transpose()[0].values
        return dataframe, label, datawithoutlabel
def HANDWRITTEN(file):
    if file == 'HandWrittenLetters.txt':
        data= pd.read_csv(file, header=None, sep=",")
        datawithoutlabel = data.iloc[1:, :]
        label = data.transpose()[0].values
        return data, label, datawithoutlabel

def ATNT400(file):
    if file == 'ATNTFaceImages400.txt':
        data= pd.read_csv(file, header=None, sep=",")
        dataframe= data.iloc[:, :400]
        datawithoutlabel = dataframe.iloc[1:, :]
        label = dataframe.transpose()[0].values
        return dataframe, label, datawithoutlabel
        
def kmeans(datanolabel, label_values, k):
    datanolabel = datanolabel.transpose()
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(datanolabel)
    labelskmeans = kmeans.labels_
    print('K means labels', labelskmeans)
    C = confusion_matrix(y_true=label_values, y_pred=labelskmeans)
    print('Confusion matrix is: ', C)
    C = C.T
    k1 = linear_assignment(-C)
    C_optimal = C[:, k1[:, 1]]
    print('re ordered matrix :', C_optimal)
    acc_opt = np.trace(C_optimal) / np.sum(C_optimal)
    accuracy = cluster_acc(label_values, labelskmeans)
    print('accuracy of k means is:', accuracy * 100)
        
def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    k1 = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in k1]) * 1.0 / y_pred.size
   
def main():
    user = input('please enter the file name: ')
    k = int(input('enter value of k: '))
    if 'ATNT' in user and k == 40:
        data, label, data_without_label = ATNT400('ATNTFaceImages400.txt')
        kmeans(data_without_label, label, k)
    elif 'Hand' in user:
        data, label, data_without_label = HANDWRITTEN('HandWrittenLetters.txt')
        kmeans(data_without_label, label, k)
    elif 'ATNT' in user:
        data, label, data_without_label = ATNT100('ATNTFaceImages400.txt')
        kmeans(data_without_label, label, k)

main()
