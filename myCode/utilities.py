import os
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from scipy.stats import norm
import math


def confusion_matrix(y_pred, y_true, method = None):
    # computer precision and recall for CNN prediction
    tn, fp, fn, tp = 0, 0, 0, 0
    #print(y_pred)
    #print(y_true)
    y_true_res = []
    y_pred_res = []
    for i in range(y_true.shape[0]):
        if y_pred[i] == 0:
            y_true_res.append(0)
            # [1,0]
            if y_true[i][0]:
                # predic 0
                tn += 1
                y_pred_res.append(0)
            else:
                # predic 1
                fp += 1
                y_pred_res.append(1)
        else:
            y_true_res.append(1)
            if y_true[i][0]:
                y_pred_res.append(0)
                fn += 1
            else:
                y_pred_res.append(1)
                tp += 1

    recall1 = tp / (fn + tp)
    precision1 = tn / (tn + fn)
    accuracy1 = (tp + tn)/(tp + tn + fp + fn)
    #f1_sc = 1 / ((1 / recall1) + (1 / precision1))
    f1_sc = 0
    if(method == 'validate'):
        print(tn, fp, fn, tp)
    return precision1, recall1, accuracy1, f1_sc


def my_Reader(path):
    datafile = os.listdir(path)
    n = len(datafile)
    selected_watch_labels = [
        'WATCH_TYPE_ORIENTATION_X', 'WATCH_TYPE_ORIENTATION_Y',
        'WATCH_TYPE_ORIENTATION_Z',
        'WATCH_TYPE_ROTATION_VECTOR_Y',

        'WATCH_TYPE_ACCELEROMETER_Y', 'WATCH_TYPE_ACCELEROMETER_Z',
        'WATCH_TYPE_GYROSCOPE_X', 'WATCH_TYPE_GYROSCOPE_Z',
        'WATCH_TYPE_GRAVITY_Y',
    ]

    label_name = ['Label']

    collection_attribues = pd.DataFrame()
    collection_label = pd.DataFrame()

    for i in range(n):
        filepath = path + datafile[i]
        file_raw = pd.read_csv(filepath)

        attributes = file_raw[selected_watch_labels]
        labels = file_raw[label_name]

        collection_attribues = pd.concat([collection_attribues, attributes], axis=0)
        collection_label = pd.concat([collection_label, labels], axis=0)

    return collection_attribues, collection_label


def make3DArray(input2dX, input2dy, input2dy1, seq_len, n_channel, n_classes):
    output3dx = np.zeros((1, n_channel, seq_len))
    output3dy = np.zeros((1, n_classes, 1))

    ite = input2dX.shape[0] // seq_len
    for i in range(ite):
        new2dx = input2dX[i:i + seq_len, ]
        new3dx = np.reshape(new2dx, newshape=(1, new2dx.shape[1], new2dx.shape[0]))

        new2dy = input2dy[i:i + seq_len, ]

        number0 = 0
        number1 = 0
        app0 = -1
        app1 = -1
        for j in range(seq_len):
            if input2dy1[i + j] == 0:
                if app0 == -1:
                    app0 = j
                number0 += 1
            else:
                if app1 == -1:
                    app1 = j
                number1 += 1

        if number1 > number0:
            newy = new2dy[app1]
        else:
            newy = new2dy[app0]

        new3dy = np.reshape(newy, newshape=(1, n_classes, 1))

        output3dx = np.concatenate((output3dx, new3dx), axis=0)
        output3dy = np.concatenate((output3dy, new3dy), axis=0)

    output1 = torch.from_numpy(output3dx[1::])
    output2 = torch.from_numpy(output3dy[1::])
    return output1, output2


def dprime(label, temp0, temp1):
    y_true = []
    y_pred = []
    for i in range(label.shape[0]):
        if label[i][0] == 1:
            y_true.append(0)
            # [1,0]
            if temp0[i]:
                # predic 0
                y_pred.append(0)
            else:
                # predic 1
                y_pred.append(1)
        else:
            y_true.append(1)
            if temp1[i]:
                y_pred.append(1)
            else:
                y_pred.append(0)

    # compute dprime for psychology usage
    Z = norm.ppf
    dprime = math.sqrt(2) * Z(roc_auc_score(y_true, y_pred))

    return dprime


def get_batch(x, y, batch_size):
    n_batches = len(x) // batch_size
    x, y = x[:n_batches * batch_size], y[:n_batches * batch_size]

    t0 = np.zeros(shape=(batch_size, 1))
    t1 = np.ones(shape=(batch_size, 1))

    # Loop over batches and yield
    for b in range(0, len(x), batch_size):
        x_ = x[b:b + batch_size]
        y_ = np.reshape(y[b:b + batch_size], newshape=(batch_size, 2))

        template0 = np.concatenate((t1, t0), axis=1)
        template1 = np.concatenate((t0, t1), axis=1)

        yield x_, y_, template0, template1
