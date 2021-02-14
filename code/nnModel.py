"""
author: Zijian Ling
date 2021.02.14
"""
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import joblib
import os
import matplotlib.pyplot as plt
import warnings
import tensorflow as tf
warnings.filterwarnings('ignore')


class Model():
    def __init__(self):

        self.dataPath = "../Combine/"
        self.datafile = os.listdir(self.dataPath)
        self.dataNumber = len(self.datafile)

        self.attri_watch_select4 = [
            'WATCH_TYPE_ORIENTATION_X', 'WATCH_TYPE_ORIENTATION_Y',
            'WATCH_TYPE_ORIENTATION_Z',
            'WATCH_TYPE_ROTATION_VECTOR_Y', 

            'WATCH_TYPE_ACCELEROMETER_Y', 'WATCH_TYPE_ACCELEROMETER_Z',
            'WATCH_TYPE_GYROSCOPE_X', 'WATCH_TYPE_GYROSCOPE_Z',
            'WATCH_TYPE_GRAVITY_Y',
        ]
        self.labelname = ['Label']

        self.collection_data = pd.DataFrame()
        self.collection_label = pd.DataFrame()


        self.ori = 0
        self.pre = 0

    def prepocessing(self):
        #print("There are total {} files, preparing for the training -------- ".format(self.dataNumber))
        #combine all data from csv files
        # for i in range(self.dataNumber):
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)


        batch_size = 200
        seq_len = 128
        learning_rate = 0.0001
        epochs = 1000

        n_classes = 2
        n_channels = 9

        for i in range(self.dataNumber):
            print(self.datafile[i])

            m_filename = self.dataPath + self.datafile[i]
            m_file = pd.read_csv(m_filename)
            m_file_1 = m_file[self.attri_watch_select4]
            m_file_label = m_file[self.labelname]

            self.collection_data = pd.concat([self.collection_data, m_file_1], axis=0)
            self.collection_label = pd.concat([self.collection_label, m_file_label], axis=0)

        print("All data has been combined together")
        print("the total data size is {}, the total lable size is {}".format(
            self.collection_data.shape, self.collection_label.shape
        ))
        
        

        local_df = pd.concat([self.collection_data, self.collection_label], axis=1)
        #local_df = local_df.sample(frac=1).reset_index(drop=True)
        
        # scaler1 = preprocessing.StandardScaler()
       
        # for i in range(len(self.attri_watch_select4)):
        #     scaler2 = np.std(local_df[[self.attri_watch_select4[i]]])
        #     local_df[[self.attri_watch_select4[i]]] = scaler1.fit_transform(local_df[[self.attri_watch_select4[i]]])
            
        #     for j in range(local_df[[self.attri_watch_select4[i]]].shape[0]):
        #         local_df[[self.attri_watch_select4[i]]] = local_df[[self.attri_watch_select4[i]]] - scaler2

        
        #split
        my_df_attris = local_df[self.attri_watch_select4].to_numpy()
        my_df_label = local_df[self.labelname].to_numpy()

        my_df_label_onhot = np.eye(n_classes)[my_df_label.reshape(-1)]
        print(my_df_label_onhot)
        print(my_df_attris.shape)
        print(my_df_label.shape)

        my_df_attris3d, my_df_label3d= self.Make3DArray(my_df_attris,my_df_label_onhot,my_df_label,
                                                        seq_len,n_channels, n_classes)

        X_, X_test, Y_, Y_test = train_test_split(my_df_attris3d, my_df_label3d, test_size=0.4)

        X_train, X_validation, Y_train, Y_validation = train_test_split(X_, Y_, test_size = 0.2)


        # Y_train = tf.one_hot(Y_train, depth= 2)
        # Y_test = tf.one_hot(Y_test, depth= 2)
        # Y_validation = tf.one_hot(Y_validation, depth= 2)

        print(X_train.shape)
        print(X_validation.shape)
        print(X_test.shape)
        


        graph = tf.Graph()

        with graph.as_default():
            inputs_ = tf.compat.v1.placeholder(tf.float32, [None, seq_len, n_channels], name = 'inputs')
            labels_ = tf.compat.v1.placeholder(tf.float32, [None, n_classes], name = 'labels')
            keep_prob_ = tf.compat.v1.placeholder(tf.float32, name = 'keep')
            learning_rate_ =tf.compat.v1.placeholder(tf.float32, name = 'learning_rate')

            temp0 = tf.compat.v1.placeholder(tf.float32,[None, n_classes], name = 'temp0')
            temp1 = tf.compat.v1.placeholder(tf.float32,[None, n_classes], name = 'temp1')
            
        with graph.as_default():
            #(batch, 128, 14) -> (batch, 64, 28)
            conv1 = tf.compat.v1.layers.conv1d(inputs= inputs_, filters= 18,
                                               kernel_size= 2, strides= 1, padding='same', activation= tf.nn.relu)

            max_pool1 = tf.compat.v1.layers.max_pooling1d(inputs = conv1, pool_size= 2, strides= 2, padding= 'same')

            conv2 = tf.compat.v1.layers.conv1d(inputs= max_pool1, filters= 36, kernel_size= 2,
                                               strides= 1, padding= 'same', activation= tf.nn.relu)

            max_pool2 = tf.compat.v1.layers.max_pooling1d(inputs= conv2, pool_size= 2, strides= 2, padding= 'same')

            conv3 = tf.compat.v1.layers.conv1d(inputs= max_pool2, filters= 72, kernel_size=2
                                               , strides= 1, padding= 'same', activation= tf.nn.relu)

            max_pool3 = tf.compat.v1.layers.max_pooling1d(inputs= conv3, pool_size= 2, strides= 2, padding= 'same')

            conv4 = tf.compat.v1.layers.conv1d(inputs= max_pool3, filters= 144, kernel_size=2, strides= 1
                                               , padding= 'same', activation= tf.nn.relu)

            max_pool4 = tf.compat.v1.layers.max_pooling1d(inputs= conv4, pool_size=2, strides= 2, padding= 'same')



        with graph.as_default():
            flat = tf.reshape(max_pool4, (-1, 8*144))
            flat = tf.nn.dropout(flat, rate= keep_prob_)

            #predictions
            logits = tf.compat.v1.layers.dense(flat, n_classes)

            corrected_pred0 = tf.equal(tf.argmax(logits, 1), tf.argmax(temp0, 1))
            corrected_pred1 = tf.equal(tf.argmax(logits, 1), tf.argmax(temp1, 1))
            #cost function and optimizer
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels_))
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate= learning_rate_).minimize(cost)

            #Accuracy
            corrected_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_,1))
            accuracy = tf.reduce_mean(tf.cast(corrected_pred, tf.float32), name='accuracy')


        if(os.path.exists('checkpoints-cnn') == False):
            print("No model")



        validation_acc = []
        validation_loss = []
        train_acc = []
        train_loss = []

        with graph.as_default():
            saver = tf.compat.v1.train.Saver()




        with tf.compat.v1.Session(graph = graph) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            iteration = 1

            #loop oover epoch
            for i in range(epochs):

                #loop over batches
                for x, y, t0, t1 in self.get_batch(X_train, Y_train, batch_size):
                    #feed dic

                    # print(t0)
                    # print(t1)
                    feed = {inputs_:x, labels_:y, keep_prob_ : 0.5, learning_rate_ : learning_rate
                            , temp0 : t0, temp1 : t1}


                    #loss
                    loss,_,acc, t0, t1 = sess.run([cost, optimizer, accuracy, corrected_pred0, corrected_pred1],
                                                  feed_dict= feed)
                    train_acc.append(acc)
                    train_loss.append(loss)

                    #print at each 10 iters
                    if(iteration%50 == 0):
                        print("Epoch: {}/{}".format(i+1, epochs),
                              "Iteration: {:d}".format(iteration),
                              "Train loss: {:6f}".format(loss),
                              "Train acc: {:6f}".format(acc))

                        # print("0 : {}".format(t0))
                        # print("1 : {}".format(t1))
                    #compute validation loss at every 10 iteration
                    if(iteration%100 == 0):
                        val_acc = []
                        val_loss = []


                        for xv, yv, yvt0, yvt1 in self.get_batch(X_validation, Y_validation, batch_size):
                            feed1 = {inputs_: xv, labels_: yv, keep_prob_: 1.0}

                            lossv, accv = sess.run([cost, accuracy], feed_dict= feed1)
                            val_acc.append(accv)
                            val_loss.append(lossv)

                        print("Epoch: {}/{}".format(i, epochs),
                                  "Iteration: {:d}".format(iteration),
                                  "Validation loss: {:6f}".format(np.mean(val_loss)),
                                  "Validation acc: {:.6f}".format(np.mean(val_acc)))

                        #Store
                        validation_acc.append(np.mean(val_acc))
                        validation_loss.append(np.mean(val_loss))

                # Iterate
                    iteration += 1

    def get_batch(self, x, y, batch_size):

        n_batches = len(x) // batch_size
        x, y = x[:n_batches * batch_size], y[:n_batches * batch_size]

        # Loop over batches and yield
        for b in range(0, len(x), batch_size):

            x_ = x[b:b + batch_size]
            y_ = np.reshape(y[b:b + batch_size], newshape= (batch_size,2))

            template0 = np.zeros(shape = (batch_size,2))
            template1 = np.ones(shape = (batch_size,2))

            yield x_, y_, template0, template1


    def Make3DArray(self, input2dX, input2dy,input2dy1,seq_len, n_channel, n_classes):

        output3dx = np.zeros((1,seq_len,n_channel))
        output3dy = np.zeros((1,1,n_classes))
        ite = input2dX.shape[0]//seq_len
        for i in range(ite):
            new2dx = input2dX[i:i + seq_len, ]
            new3dx = np.reshape(new2dx, newshape= (1,new2dx.shape[0], new2dx.shape[1]))

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


            new3dy = np.reshape(newy, newshape = (1,1, n_classes))

            output3dx = np.concatenate((output3dx, new3dx), axis= 0)
            output3dy = np.concatenate((output3dy, new3dy), axis= 0)


        return output3dx[1: : ], output3dy[1: : ]




    def main(self):
        self.prepocessing()
      
        #self.testing()
if __name__ == '__main__':
    Solution = Model()
    Solution.main()
