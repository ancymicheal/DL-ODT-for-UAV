
# Imports
import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import cv2
import numpy as np
import os.path
import time
import random
import sys
sys.path.append("./ROLO/utils")
import ROLO_utils as utils

class ROLO_TF:
    disp_console = True
    restore_weights = True  # False

    # YOLO parameters
    fromfile = None
    tofile_img = 'test/output.jpg'
    tofile_txt = 'test/output.txt'
    imshow = True
    filewrite_img = False
    filewrite_txt = False
    disp_console = True
    yolo_weights_file = './ROLO/weights/YOLO_small.ckpt'
    alpha = 0.1
    threshold = 0.2
    iou_threshold = 0.5
    num_class = 20
    num_box = 2
    grid_size = 7
    classes =  ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train" ,"tvmonitor"]
    w_img, h_img = [352, 240]

    # ROLO Network Parameters
    rolo_weights_file = 'null'
    # rolo_weights_file = '/u03/Guanghan/dev/ROLO-dev/model_dropout_30.ckpt'
    lstm_depth = 3
    num_steps = 3  # number of frames as an input sequence
    num_feat = 4096
    num_predict = 6 # final output of LSTM 6 loc parameters
    num_gt = 4
    num_input = num_feat + num_predict # data input: 4096+6= 5002
    # testing_iters = 406
    # ROLO Parameters
    batch_size = 1
    display_step = 1

    # tf Graph input
    x = tf.placeholder("float32", [None, num_steps, num_input])
    istate = tf.placeholder("float32", [None, 2* num_input])  # state & cell => 2x num_input
    y = tf.placeholder("float32", [None, num_gt])

    # Define weights
    weights = {
        'out': tf.Variable(tf.random_normal([num_input, num_predict]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([num_predict]))
    }

    def __init__(self, argvs=[]):
        print("ROLO init")
        self.ROLO(argvs)

    def LSTM_single(self, name, _X, _istate, _weights, _biases):
        # with tf.device('/gpu:0'):
        # input shape: (batch_size, n_steps, n_input)
        _X = tf.transpose(_X, [1, 0, 2])  # permute num_steps and batch_size
        # Reshape to prepare input to hidden activation
        _X = tf.reshape(_X, [self.num_steps * self.batch_size, self.num_input])  # (num_steps*batch_size, num_input)
        # Split data because rnn cell needs a list of inputs for the RNN inner loop
        _X = tf.split(0, self.num_steps, _X)  # n_steps * (batch_size, num_input)

        cell = tf.nn.rnn_cell.LSTMCell(self.num_input, self.num_input)
        state = _istate
        for step in range(self.num_steps):
            outputs, state = tf.nn.rnn(cell, [_X[step]], state)
            tf.get_variable_scope().reuse_variables()
        return outputs

    # Experiment with dropout
    def dropout_features(self, feature, prob):
        num_drop = int(prob * 4096)
        drop_index = random.sample(xrange(4096), num_drop)
        for i in range(len(drop_index)):
            index = drop_index[i]
            feature[index] = 0
        return feature

    '''---------------------------------------------------------------------------------------'''

    def build_networks(self):
        if self.disp_console: print "Building ROLO graph..."

        # Build rolo layers
        self.lstm_module = self.LSTM_single('lstm_test', self.x, self.istate, self.weights, self.biases)
        self.ious = tf.Variable(tf.zeros([self.batch_size]), name="ious")
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())
        self.saver = tf.train.Saver()
        # self.saver.restore(self.sess, self.rolo_weights_file)
        if self.disp_console: print "Loading complete!" + '\n'

    def testing(self, x_path, y_path):
        total_loss = 0
        # Use rolo_input for LSTM training
        pred = self.LSTM_single('lstm_train', self.x, self.istate, self.weights, self.biases)
        # print("pred: ", pred)
        self.pred_location = pred[0][:, 4097:4101]
        self.correct_prediction = tf.square(self.pred_location - self.y)
        # print("self.correct_prediction: ", self.correct_prediction)
        self.accuracy = tf.reduce_mean(self.correct_prediction) * 100
        init = tf.initialize_all_variables()
        # Launch the graph
        with tf.Session() as sess:

            if (self.restore_weights == True):
                sess.run(init)
                self.saver.restore(sess, self.rolo_weights_file)
                print "Loading complete!" + '\n'
            else:
                sess.run(init)

            id = 0  # don't change this
            total_time = 0.0
            # id= 1

            # Keep training until reach max iterations
            while id < self.testing_iters - self.num_steps:
                # Load training data & ground truth
                batch_xs = self.rolo_utils.load_yolo_output_test(x_path, self.batch_size, self.num_steps,
                                                                 id)  # [num_of_examples, num_input] (depth == 1)

                # Apply dropout to batch_xs
                # for item in range(len(batch_xs)):
                #    batch_xs[item] = self.dropout_features(batch_xs[item], 0.4)

                batch_ys = self.rolo_utils.load_rolo_gt_test(y_path, self.batch_size, self.num_steps, id)
                batch_ys = utils.locations_from_0_to_1(self.w_img, self.h_img, batch_ys)

                # Reshape data to get 3 seq of 5002 elements
                batch_xs = np.reshape(batch_xs, [self.batch_size, self.num_steps, self.num_input])
                batch_ys = np.reshape(batch_ys, [self.batch_size, 4])
                # print("Batch_ys: ", batch_ys)

                start_time = time.time()
                pred_location = sess.run(self.pred_location, feed_dict={self.x: batch_xs, self.y: batch_ys,
                                                                        self.istate: np.zeros(
                                                                            (self.batch_size, 2 * self.num_input))})
                cycle_time = time.time() - start_time
                total_time += cycle_time

                # Save pred_location to file
                utils.save_rolo_output_test(self.output_path, pred_location, id, self.num_steps, self.batch_size)

                # sess.run(optimizer, feed_dict={self.x: batch_xs, self.y: batch_ys, self.istate: np.zeros((self.batch_size, 2*self.num_input))})

                if id % self.display_step == 0:
                    # Calculate batch loss
                    loss = sess.run(self.accuracy, feed_dict={self.x: batch_xs, self.y: batch_ys, self.istate: np.zeros(
                        (self.batch_size, 2 * self.num_input))})
                    # print "Iter " + str(id*self.batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) #+ "{:.5f}".format(self.accuracy)
                    total_loss += loss
                id += 1
                # print(id)

            # print "Testing Finished!"
            avg_loss = total_loss / id
            print "Avg loss: " + str(avg_loss)
            print "Time Spent on Tracking: " + str(total_time)
            print "fps: " + str(id / total_time)
            # save_path = self.saver.save(sess, self.rolo_weights_file)
            # print("Model saved in file: %s" % save_path)

        return None

    def ROLO(self, argvs):
        self.rolo_utils = utils.ROLO_utils()
        self.rolo_utils.loadCfg()
        self.params = self.rolo_utils.params

        arguments = self.rolo_utils.argv_parser(argvs)

        if self.rolo_utils.flag_train is True:
            self.training(utils.x_path, utils.y_path)
        elif self.rolo_utils.flag_track is True:
            self.build_networks()
            self.track_from_file(utils.file_in_path)
        elif self.rolo_utils.flag_detect is True:
            self.build_networks()
            self.detect_from_file(utils.file_in_path)
        else:
            print "Default: running ROLO test."
            self.build_networks()
	
        folders = os.listdir("./ROLO/DATA/")
	
        for folder_name in folders:
            output_path = os.path.join('ROLO/DATA', folder_name, 'rolo_out_test/')
            y_path = os.path.join('ROLO/DATA', folder_name, 'groundtruth_rect.txt')
            x_path = os.path.join('ROLO/DATA', folder_name, 'yolo_out/')
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            self.output_path = output_path
        # print(output_path)
	        print("******************************************")
            test_iter_path = os.path.join('ROLO/DATA', folder_name, 'img/')
            test_iter_len = next(os.walk(test_iter_path))[2]  # dir is your directory path as string
            self.testing_iters = len(test_iter_len)
            self.rolo_weights_file = './ROLO/output/ROLO_model/model_step1_exp2.ckpt'

            self.num_steps = 3  # number of frames as an input sequence
        # print("TESTING ROLO on video sequence: ", sequence_name)
            print("TESTING ROLO on video sequence: ", folder_name)
       	    self.testing(x_path, y_path)

    '''----------------------------------------main-----------------------------------------------------'''


def main(argvs):
    ROLO_TF(argvs)


if __name__ == '__main__':
    main(' ')
