import os
import torch
import torch.nn as nn
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from model_cnn import CNNmodel
from utilities import my_Reader
from utilities import get_batch
from utilities import make3DArray
from utilities import confusion_matrix


class Model():
    def __init__(self):
        self.selected_features = [
            'WATCH_TYPE_ORIENTATION_X', 'WATCH_TYPE_ORIENTATION_Y',
            'WATCH_TYPE_ORIENTATION_Z',
            'WATCH_TYPE_ROTATION_VECTOR_Y',

            'WATCH_TYPE_ACCELEROMETER_Y', 'WATCH_TYPE_ACCELEROMETER_Z',
            'WATCH_TYPE_GYROSCOPE_X', 'WATCH_TYPE_GYROSCOPE_Z',
            'WATCH_TYPE_GRAVITY_Y',
        ]

        self.selected_label = ['Label']

        self.cnn_batchSize = 200
        self.cnn_seqLen = 128
        #self.cnn_learningRate = 0.00000000000001
        self.cnn_learningRate = 0.000001
        self.cnn_epochs = 10
        self.n_classes = 2
        self.n_channels = 9
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)
        # model instance
        self.cnnModel = CNNmodel()
        self.cnnModel.to(self.device)
        self.optimizer = torch.optim.AdamW(self.cnnModel.parameters(), lr=self.cnn_learningRate)
        #self.criterion = nn.BCEWithLogitsLoss()
        self.criterion = nn.CrossEntropyLoss()
        datapath = "../Combine/"
        attributes, labels = my_Reader(datapath)

        local_df = pd.concat([attributes, labels], axis=1)

        # scale data using sklearn library
        scaler = preprocessing.MaxAbsScaler()
        for i in range(len(self.selected_features)):
            local_df[[self.selected_features[i]]] = scaler.fit_transform(local_df[[self.selected_features[i]]])

        my_df_features = local_df[self.selected_features].to_numpy()
        my_df_label = local_df[self.selected_label].to_numpy()
        my_df_label_onhot = np.eye(self.n_classes)[my_df_label.reshape(-1)]

        my_df_features3d, my_df_label_3d = make3DArray(my_df_features, my_df_label_onhot, my_df_label,
                                                       self.cnn_seqLen, self.n_channels, self.n_classes)

        self.X_train, self.X_validate, self.Y_train, self.Y_validate = train_test_split(my_df_features3d,
                                                                                        my_df_label_3d, test_size=0.2)

    def train(self, epoch, model):
        model.train()
        iteration = 0
        loss_ = 0
        precision_, recall_, accuracy_ = 0, 0, 0
        for x, y, t0, t1 in get_batch(self.X_train, self.Y_train, self.cnn_batchSize):
            x = x.to(self.device)
            y = y.to(self.device, torch.int64)
            iteration += 1
            output = model(x)
            output = output.squeeze(dim=-1)
            yy = torch.argmax(y, dim=1)
            # print("ouput shape {}, output type = {}".format(output.shape, type(output)))
            # print("yy shape {}, y type = {}".format(yy.shape, type(yy)))
            loss = self.criterion(output, yy)
            loss.backward()
            nn.utils.clip_grad_norm(self.cnnModel.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
            loss_ += loss.item()
            output = torch.argmax(output, 1)
            precision, recall, accuracy2, f1_sc = confusion_matrix(output.cpu().detach().numpy(),
                                                                   y.cpu().detach().numpy(), method='train')
            precision_ += precision
            recall_ += recall
            accuracy_ += accuracy2
        print('Epoch {}, train loss = {}'.format(epoch, loss_ / iteration))
        print('Epoch {}, precision = {}, recall = {}, accuracy = {}'.format(epoch, precision_/iteration,
                                                                            recall_/iteration, accuracy_/iteration))
        print()
        return loss_ / iteration

    def validate(self, epoch, model):
        model.eval()
        iteration = 0
        loss_ = 0
        precision_, recall_, accuracy_ = 0,0,0
        with torch.no_grad():
            for xv, yv, yvt0, yvt1 in get_batch(self.X_validate, self.Y_validate, self.cnn_batchSize):
                xv = xv.to(self.device)
                yv = yv.to(self.device, torch.int64)
                output = model(xv)
                iteration += 1

                output = output.squeeze(dim=-1)
                yy = torch.argmax(yv, dim=1)
                loss = self.criterion(output, yy)
                loss_ += loss.item()

                output = torch.argmax(output, 1)
                precision, recall, accuracy2, f1_sc = confusion_matrix(output.cpu().detach().numpy(),
                                                                       yv.cpu().detach().numpy())
                precision_ += precision
                recall_ += recall
                accuracy_ += accuracy2
        print('Epoch {}, validation loss = {}'.format(epoch, loss_ / iteration))
        return precision_/iteration, recall_/iteration, accuracy_/iteration, f1_sc
    def visualise(self):
        data = [l.strip().split('\t') for l in open('log/analysis.txt','r').readlines()]

        
    def main(self):
        print("Begin---------")

        # create file for log
        if (not os.path.exists("./log")):
            os.mkdir("./log")

        if not os.path.exists('./log/analysis.txt'):
            os.mknod('./log/analysis.txt')
        else:
            os.remove('./log/analysis.txt')
            os.mknod('./log/analysis.txt')
        for epoch in range(self.cnn_epochs):
            loss = self.train(epoch + 1, self.cnnModel)

            if ((epoch + 1) % 50 == 0):
                precision, recall, accuracy, f1_sc = self.validate(epoch + 1, self.cnnModel)
                print('Epoch {}, precision = {}, recall = {}, accuracy = {}'.format(epoch+1, precision,
                                                                                    recall, accuracy))
                print()
                #self.validate(epoch + 1, self.cnnModel)
                # result = [precision, recall, accuracy, f1_sc]
                # result = [str(i) for i in result]
                #
                # with open('./log/analysis.txt', 'a') as f:
                #     f.write('\t'.join(result) + '\n')
                # f.close()


if __name__ == '__main__':
    Solution = Model()
    Solution.main()
