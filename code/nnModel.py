"""
author: Zijian Ling
date 2021.02.21
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
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
        learning_rate = 0.000000000001
        epochs = 10000

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
        
        scaler1 = preprocessing.StandardScaler()
        scaler1 = preprocessing.MaxAbsScaler()
        #scaler1 = preprocessing.MinMaxScaler()
        for i in range(len(self.attri_watch_select4)):
            local_df[[self.attri_watch_select4[i]]] = scaler1.fit_transform(local_df[[self.attri_watch_select4[i]]])
            
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
          

            is_training = tf.compat.v1.placeholder(tf.bool)

        with graph.as_default():
            #(batch, 128, 14) -> (batch, 64, 28)
            conv1 = tf.compat.v1.layers.conv1d(inputs= inputs_, filters= 18,
                                               kernel_size= 2, strides= 1, padding='same', activation= tf.nn.tanh,
                                               kernel_initializer = tf.compat.v1.truncated_normal_initializer(stddev=0.1),
                                               )
            bn1 = tf.compat.v1.layers.batch_normalization(conv1, training=is_training, name='bn1')
            max_pool1 = tf.compat.v1.layers.max_pooling1d(inputs = bn1, pool_size= 2, strides= 2, padding= 'same')

            conv2 = tf.compat.v1.layers.conv1d(inputs= max_pool1, filters= 36, kernel_size= 2,
                                               strides= 1, padding= 'same', activation= tf.nn.relu,
                                               kernel_initializer = tf.compat.v1.truncated_normal_initializer(stddev=0.1))
            bn2 = tf.compat.v1.layers.batch_normalization(conv2, training=is_training, name='bn2')
            max_pool2 = tf.compat.v1.layers.max_pooling1d(inputs= bn2, pool_size= 2, strides= 2, padding= 'same')


            conv3 = tf.compat.v1.layers.conv1d(inputs= max_pool2, filters= 72, kernel_size=2
                                               , strides= 1, padding= 'same', activation= tf.nn.relu,
                                               kernel_initializer = tf.compat.v1.truncated_normal_initializer(stddev=0.1))
            bn3 = tf.compat.v1.layers.batch_normalization(conv3, training=is_training, name='bn3')
            max_pool3 = tf.compat.v1.layers.max_pooling1d(inputs= bn3, pool_size= 2, strides= 2, padding= 'same')


            conv4 = tf.compat.v1.layers.conv1d(inputs= max_pool3, filters= 144, kernel_size=2, strides= 1
                                               , padding= 'same', activation= tf.nn.relu,
                                               kernel_initializer = tf.compat.v1.truncated_normal_initializer(stddev=0.1))
            bn4 = tf.compat.v1.layers.batch_normalization(conv4, training=is_training, name='bn4')
            max_pool4 = tf.compat.v1.layers.max_pooling1d(inputs= bn4, pool_size=2, strides= 2, padding= 'same')



        with graph.as_default():
            flat = tf.reshape(max_pool4, (-1, 8*144))
            flat = tf.nn.dropout(flat, rate= keep_prob_)


            #predictions
            logits = tf.compat.v1.layers.dense(flat, n_classes)

            corrected_pred0 = tf.equal(tf.argmax(logits, 1), tf.argmax(temp0, 1))
            corrected_pred1 = tf.equal(tf.argmax(logits, 1), tf.argmax(temp1, 1))
            #cost function and optimize
            #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels_))
            #cost = tf.reduce_mean(-tf.reduce_sum(labels_*tf.math.log(logits + 1e-10)))
            cost = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true=labels_, y_pred=logits, from_logits=True))
            with tf.control_dependencies(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)):

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
            
            precision_list = []
            recall_list = []
            #loop oover epoch   
            for i in range(epochs):

                #loop over batches
                for x, y, t0, t1 in self.get_batch(X_train, Y_train, batch_size):
                    #feed dic

                    # print(t0)
                    # print(t1)
                    # feed = {inputs_:x, labels_:y, keep_prob_ : 0.5, learning_rate_ : learning_rate
                    #         , temp0 : t0, temp1 : t1}

                    feed_pre = {inputs_:x, labels_:y, keep_prob_ : 0.5, learning_rate_ : learning_rate
                            , temp0 : t0, temp1 : t1, is_training:True}
                    #loss
                    # loss,_,acc, c0, c1 = sess.run([cost, optimizer, accuracy, corrected_pred0, corrected_pred1,
                    #                                         ],
                    #                               feed_dict= feed)


                    # train_acc.append(acc)
                    # train_loss.append(loss)
                
                    sess.run(optimizer, feed_dict = feed_pre)
                            
                    #print at each 50 iters
                    if(iteration%200 == 0):

                        feed = {inputs_:x, labels_:y, keep_prob_ : 0.5, 
                            temp0 : t0, temp1 : t1, is_training : False}

                        loss,acc, c0, c1 = sess.run([cost, accuracy, corrected_pred0, corrected_pred1,
                                                            ],
                                                  feed_dict= feed)

                        print("Epoch: {}/{}".format(i+1, epochs),
                              "Iteration: {:d}".format(iteration),
                              "Train loss: {:6f}".format(loss),
                              "Train acc: {:6f}".format(acc))

                        _, _ = self.confusion_matrix(y,c0, c1,"train")
                        train_acc.append(acc)
                        train_loss.append(loss)
                    #compute validation loss at every 100 iteration
                    if(iteration%1000 == 0):
                        val_acc = []
                        val_loss = []


                        for xv, yv, yvt0, yvt1 in self.get_batch(X_validation, Y_validation, batch_size):
                            feed1 = {inputs_: xv, labels_: yv, keep_prob_: 0.5,
                                    temp0: yvt0, temp1:yvt1, is_training : False
                                    }

                            lossv, accv, c0v, c1v = sess.run([cost, accuracy, corrected_pred0, 
                                                                    corrected_pred1], feed_dict= feed1)

                            print(lossv)
                            val_acc.append(accv)
                            val_loss.append(lossv)

                        print("Epoch: {}/{}".format(i, epochs),
                                  "Iteration: {:d}".format(iteration),
                                  "Validation loss: {:6f}".format(np.mean(val_loss)),
                                  "Validation acc: {:.6f}".format(np.mean(val_acc)))
                        precision, recall = self.confusion_matrix(yv,c0v, c1v, "validation")
                        #Store
                        validation_acc.append(np.mean(val_acc))
                        validation_loss.append(np.mean(val_loss))

                        precision_list.append(precision)
                        recall_list.append(recall)
                # Iterate
                    iteration += 1

            print("average precision in validation: {:6f}".format(np.mean(precision_list)))
            print("average recall in validation: {:6f}".format(np.mean(recall_list)))
    def get_batch(self, x, y, batch_size):

        n_batches = len(x) // batch_size
        x, y = x[:n_batches * batch_size], y[:n_batches * batch_size]

        t0 = np.zeros(shape = (batch_size,1))
        t1 = np.ones(shape = (batch_size,1))

        # Loop over batches and yield
        for b in range(0, len(x), batch_size):

            x_ = x[b:b + batch_size]
            y_ = np.reshape(y[b:b + batch_size], newshape= (batch_size,2))

            template0 = np.concatenate((t1,t0), axis=1)
            template1 = np.concatenate((t0,t1), axis=1)
     
          
            yield x_, y_, template0, template1


    def Make3DArray(self, input2dX, input2dy,input2dy1,seq_len, n_channel, n_classes):

        output3dx = np.zeros((1,seq_len,n_channel))
        output3dy = np.zeros((1,1,n_classes))
        outputy0 = np.zeros((1,1,n_classes))
        outputy1 = np.zeros((1,1,n_classes))


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


    def confusion_matrix(self, label, temp0, temp1, method = None):
        #computer precision and recall for CNN prediction
        tn = 0
        fp = 0
        fn = 0
        tp = 0
        for i in range(label.shape[0]):
            if label[i][0] == 1:
                #[1,0]
                if temp0[i]:
                    #predic 0
                    tn += 1
                else:
                    #predic 1
                    fp += 1
            else:
                if temp1[i]:
                    tp += 1
                else:
                    fn += 1

        recall1 = tp/(fn+tp)
        precision1 = tn/(tn + fn)
        print(method + '----------')
        print("percent true {}".format(recall1))
        print("percent false {}".format(precision1))
        print("f1 score: {}".format(1/((1/recall1) + (1/precision1))))
        print()
        
        return precision1, recall1
    def main(self):
        self.prepocessing()
      
        #self.testing()
if __name__ == '__main__':
    Solution = Model()
    Solution.main()

