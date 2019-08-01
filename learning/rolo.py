import os
import sys
import time
import tkMessageBox

import numpy as np
import tensorflow as tf

sys.path.append("./rolo/utils")
import ROLO_utils as utils


def get_epochs(epoch):
    return len(list(os.listdir("./rolo/data/"))) * epoch


class ROLO:
    restore_weights = True
    num_steps = 1
    num_feat = 4096
    num_predict = 6
    num_input = num_feat + num_predict
    batch_size = 1
    num_gt = 4
    yolo_weights_file = "./rolo/weights/YOLO_small.ckpt"
    rolo_weights_file = './rolo/output/rolo_model/model_step1_exp2_new.ckpt'
    rolo_meta_file = './rolo/output/rolo_model/model_step1_exp2_new.ckpt.meta'

    # tf Graph input
    istate = tf.compat.v1.placeholder("float32", [None, 2 * num_input])  # state & cell => 2x num_input
    x = tf.compat.v1.placeholder("float32", [None, num_steps, num_input])
    y = tf.compat.v1.placeholder("float32", [None, num_gt])
    seq_len = tf.compat.v1.placeholder("float32", [None])

    # Define weights
    weights = {
        'out': tf.Variable(tf.random_normal([num_input, num_predict]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([num_predict]))
    }

    rolo_utils = utils.ROLO_utils()
    w_img, h_img = [352, 240]
    display_step = 1

    def __init__(self):
        pass

    def train(self, epoch):
        print("Training ROLO..")
        with tf.variable_scope('for_reuse_scope'):
            self.build_networks()
            pred = self.LSTM_single('lstm_train', self.x, self.istate, self.weights, self.biases)
        self.pred_location = pred[0][:, 4097:4101]
        self.correct_prediction = tf.square(self.pred_location - self.y)
        self.accuracy = tf.reduce_mean(self.correct_prediction) * 100
        self.learning_rate = 0.00001
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate) \
            .minimize(self.accuracy)  # Adam Optimizer

        # calculate no. of epoches
        # todo, get epoch input from user
        self.epochs = get_epochs(epoch)

        init = tf.initialize_all_variables()
        with tf.Session() as sess:
            if self.restore_weights:
                sess.run(init)
                self.saver.restore(sess, self.rolo_weights_file)
                print "Loading complete!" + '\n'
            else:
                sess.run(init)

            for epoch in range(self.epochs):
                folders = os.listdir("./rolo/data/")
                for folder_name in folders:
                    x_path = os.path.join('./rolo/data', folder_name, 'yolo_out/')
                    y_path = os.path.join('./rolo/data', folder_name, 'groundtruth_rect.txt')
                    img_path = os.path.join('./rolo/data', folder_name, 'img/')
                    train_iter_len = next(os.walk(img_path))[2]
                    training_iters = len(train_iter_len)
                    training_iters1 = int(round(training_iters / 3) - 5)
                    output_path = os.path.join('rolo/data', folder_name, 'rolo_out_train/')
                    if not os.path.exists(output_path):
                        os.makedirs(output_path)
                    self.output_path = output_path

                    i = 1
                    total_loss = 0

                    # Keep training until reach max iterations
                    while i < training_iters1 - self.num_steps:
                        # Load training data & ground truth
                        batch_xs = self.rolo_utils.load_yolo_output_test(
                            x_path, self.batch_size, self.num_steps, i
                        )  # [num_of_examples, num_input] (depth == 1)

                        batch_ys = self.rolo_utils.load_rolo_gt_test(
                            y_path, self.batch_size, self.num_steps, i
                        )

                        batch_ys = utils.locations_from_0_to_1(self.w_img, self.h_img, batch_ys)

                        # Reshape data to get 3 seq of 5002 elements
                        batch_xs = np.reshape(batch_xs, [self.batch_size, self.num_steps, self.num_input])
                        batch_ys = np.reshape(batch_ys, [self.batch_size, 4])
                        print("Batch_ys: ", batch_ys)

                        pred_location = sess.run(self.pred_location, feed_dict={self.x: batch_xs, self.y: batch_ys,
                                                                                self.istate: np.zeros(
                                                                                    (self.batch_size,
                                                                                     2 * self.num_input))})
                        print("ROLO Pred: ", pred_location)

                        print(
                            "ROLO Pred in pixel: ",
                            pred_location[0][0] * self.w_img, pred_location[0][1] * self.h_img,
                            pred_location[0][2] * self.w_img, pred_location[0][3] * self.h_img
                        )

                        utils.save_rolo_output_test(self.output_path, pred_location, i, self.num_steps, self.batch_size)

                        sess.run(self.optimizer, feed_dict={self.x: batch_xs, self.y: batch_ys,
                                                            self.istate: np.zeros(
                                                                (self.batch_size, 2 * self.num_input))})
                        if i % self.display_step == 0:
                            # Calculate batch loss
                            loss = sess.run(self.accuracy, feed_dict={self.x: batch_xs, self.y: batch_ys,
                                                                      self.istate: np.zeros(
                                                                          (self.batch_size, 2 * self.num_input))})
                        print(
                                "Iter " + str(i * self.batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss)
                        )  # + "{:.5f}".format(self.accuracy)

                        total_loss += loss
                        i += 1
                    avg_loss = total_loss / i
                    print "Avg loss: " + folder_name + ": " + str(avg_loss)
                    save_path = self.saver.save(sess, self.rolo_weights_file)
                    print("Model saved in file: %s" % save_path)
                    print("Average Loss", "Avg loss: " + folder_name + ": " + str(avg_loss))
            tkMessageBox.showinfo("Success", "Model successfully \n Saved")

        # close graph and session
        sess.close()

    def test(self):
        # todo, check if the network is built already, else build the network
        folders = os.listdir("./rolo/data/")
        for folder_name in folders:
            output_path = os.path.join('./rolo/data', folder_name, 'rolo_out_test/')
            y_path = os.path.join('./rolo/data', folder_name, 'groundtruth_rect.txt')
            x_path = os.path.join('./rolo/data', folder_name, 'yolo_out/')
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            self.output_path = output_path
            test_iter_path = os.path.join('./rolo/data', folder_name, 'img/')
            test_iter_len = next(os.walk(test_iter_path))[2]  # dir is your directory path as string
            self.testing_iters = len(test_iter_len)
            self.rolo_weights_file = './rolo/output/rolo_model/model_step1_exp2_new.ckpt'
            self.num_steps = 3  # number of frames as an input sequence
            print("TESTING ROLO on video sequence: ", folder_name)
            self.testing(x_path, y_path)

    def testing(self, x_path, y_path):
        total_loss = 0
        # Use rolo_input for LSTM training
        pred = self.LSTM_single('lstm_train', self.x, self.istate, self.weights, self.biases)
        # print("pred: ", pred)
        self.pred_location = pred[0][:, 4097:4101]
        # print("pred_location: ", self.pred_location)
        # print("self.y: ", self.y)
        self.correct_prediction = tf.square(self.pred_location - self.y)
        # print("self.correct_prediction: ", self.correct_prediction)
        self.accuracy = tf.reduce_mean(self.correct_prediction) * 100
        # print("self.accuracy: ", self.accuracy)
        # optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.accuracy) # Adam Optimizer

        # Initializing the variables
        init = tf.initialize_all_variables()
        # Launch the graph
        with tf.Session() as sess:
            if self.restore_weights:
                sess.run(init)
                self.saver.restore(sess, self.rolo_weights_file)
                print "Loading complete!" + '\n'
            else:
                sess.run(init)

            i = 0  # don't change this
            print(i)
            total_time = 0.0

            # Keep training until reach max iterations
            while i < self.testing_iters - self.num_steps:
                # Load training data & ground truth
                batch_xs = self.rolo_utils.load_yolo_output_test(
                    x_path, self.batch_size, self.num_steps, i
                )  # [num_of_examples, num_input] (depth == 1)

                # Apply dropout to batch_xs
                # for item in range(len(batch_xs)):
                #    batch_xs[item] = self.dropout_features(batch_xs[item], 0.4)

                batch_ys = self.rolo_utils.load_rolo_gt_test(y_path, self.batch_size, self.num_steps, i)
                batch_ys = utils.locations_from_0_to_1(self.w_img, self.h_img, batch_ys)

                # Reshape data to get 3 seq of 5002 elements
                batch_xs = np.reshape(batch_xs, [self.num_steps, self.batch_size, self.num_input])
                batch_ys = np.reshape(batch_ys, [self.batch_size, 4])
                # print("Batch_ys: ", batch_ys)

                start_time = time.time()
                pred_location = sess.run(self.pred_location, feed_dict={
                    self.x: batch_xs, self.y: batch_ys,
                    self.istate: np.zeros((self.batch_size, 2 * self.num_input))
                })
                cycle_time = time.time() - start_time
                total_time += cycle_time

                # Save pred_location to file
                utils.save_rolo_output_test(self.output_path, pred_location, i, self.num_steps, self.batch_size)

                if i % self.display_step == 0:
                    # Calculate batch loss
                    loss = sess.run(self.accuracy, feed_dict={self.x: batch_xs, self.y: batch_ys, self.istate: np.zeros(
                        (self.batch_size, 2 * self.num_input))})
                    total_loss += loss
                i += 1

            # print "Testing Finished!"
            avg_loss = total_loss / i
            print "Avg loss: " + str(avg_loss)
            print "Time Spent on Tracking: " + str(total_time)
            print "fps: " + str(i / total_time)
            # save_path = self.saver.save(sess, self.rolo_weights_file)
            # print("Model saved in file: %s" % save_path)

            # close graph and session
            sess.close()

    def build_networks(self):
        print("Building ROLO graph...")
        # tf.reset_default_graph()
        self.lstm_module = self.LSTM_single('lstm_test', self.x, self.istate, self.weights, self.biases)
        self.ious = tf.Variable(tf.zeros([self.batch_size]), name="ious")
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

        # self.saver = tf.compat.v1.train.import_meta_graph(self.rolo_meta_file)
        self.saver = tf.compat.v1.train.Saver()
        # self.saver.restore(self.sess, self.rolo_weights_file)

        print("Loading complete!")

    def LSTM_single(self, name, _X, _istate, _weights, _biases):
        # input shape: (batch_size, n_steps, n_input)
        _X = tf.transpose(_X, [1, 0, 2])  # permute num_steps and batch_size
        # Reshape to prepare input to hidden activation
        _X = tf.reshape(_X, [self.num_steps * self.batch_size, self.num_input])  # (num_steps*batch_size, num_input)
        # Split data because rnn cell needs a list of inputs for the RNN inner loop
        _X = tf.split(_X, self.num_steps, 0)  # n_steps * (batch_size, num_input)
        print("_X: ", _X)

        cell = tf.nn.rnn_cell.LSTMCell(self.num_input, self.num_input, state_is_tuple=False)
        state = _istate
        for step in range(self.num_steps):
            outputs, state = tf.nn.static_rnn(cell, [_X[step]], state)
            tf.get_variable_scope().reuse_variables()
        return outputs


if __name__ == "__main__":
    ROLO().test()
