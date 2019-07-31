import os
import time

import PIL
import cv2
import numpy as np
import tensorflow as tf
from PIL import ImageTk
from pathlib import Path
from Tkinter import *


def pooling_layer(idx, inputs, size, stride):
    print('Layer  %d : Type = Pool, Size = %d * %d, Stride = %d' % (idx, size, size, stride))
    return tf.nn.max_pool2d(inputs, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding='SAME',
                            name=str(idx) + '_pool')


def load_folder(path):
    return sorted([os.path.join(path, fn) for fn in next(os.walk(path))[2]])


def load_dataset_gt(gt_file):
    return open(gt_file, "r") \
        .read() \
        .split('\n')


def file_to_img(filepath):
    return cv2.imread(filepath)


def find_gt_location(lines, id):
    line = lines[id]
    elems = line.split('\t')  # for gt type 2
    if len(elems) < 4:
        elems = line.split(',')  # for gt type 1
    x1 = elems[0]
    y1 = elems[1]
    w = elems[2]
    h = elems[3]
    return [int(x1), int(y1), int(w), int(h)]


def iou(box1, box2):
    tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - max(box1[0] - 0.5 * box1[2],
                                                                     box2[0] - 0.5 * box2[2])
    lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - max(box1[1] - 0.5 * box1[3],
                                                                     box2[1] - 0.5 * box2[3])
    if tb < 0 or lr < 0:
        intersection = 0
    else:
        intersection = tb * lr
    return intersection / (box1[2] * box1[3] + box2[2] * box2[3] - intersection)


def debug_location(img, location, controller):
    img_cp = img.copy()
    x = int(location[1])
    y = int(location[2])
    w = int(location[3]) // 2
    h = int(location[4]) // 2
    cv2.rectangle(img_cp, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
    cv2.rectangle(img_cp, (x - w, y - h - 20), (x + w, y - h), (125, 125, 125), -1)
    cv2.putText(img_cp, str(location[0]) + ' : %.2f' % location[5], (x - w + 5, y - h - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    # cv2.imshow('YOLO_small detection', img_cp)
    # cv2.waitKey(1)

    controller.tk_images.append(PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img_cp)))
    controller.main_panel.after(100, controller.show_location)


def debug_gt_location(img, location):
    img_cp = img.copy()
    x = int(location[0])
    y = int(location[1])
    w = int(location[2])
    h = int(location[3])
    # cv2.rectangle(img_cp, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # cv2.imshow('gt', img_cp)
    # cv2.waitKey(1)


def location_from_0_to_1(wid, ht, location):
    location[1] /= wid
    location[2] /= ht
    location[3] /= wid
    location[4] /= ht
    return location


def save_yolo_output(out_fold, yolo_output, filename):
    name_no_ext = os.path.splitext(filename)[0]
    output_name = name_no_ext
    path = os.path.join(out_fold, output_name)
    np.save(path, yolo_output)


class YoloTf:
    alpha = 0.1
    threshold = 0.08
    iou_threshold = 0.5
    num_feat = 4096
    num_predict = 6  # final output of LSTM 6 loc parameters

    weights_file = './rolo/weights/YOLO_small.ckpt'
    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    def __init__(self):
        self.build_networks()

    def prepare_training_data(self, controller):
        # [or]prepare_training_data(self, list_file, gt_file, out_fold):
        """
        Pass the data through YOLO, and get the fc_17 layer as features, and get the fc_19 layer as locations
        Save the features and locations into file for training LSTM
        :param controller:
        """
        # Reshape the input image

        base_path = Path("./rolo/data/")

        avg_loss = 0
        total = 0
        total_time = 0

        for entry in base_path.iterdir():
            if entry.is_dir():
                img_fold = os.path.join('./rolo/data', entry.name, 'img/')
                gt_file = os.path.join('./rolo/data', entry.name, 'groundtruth_rect.txt')
                out_fold = os.path.join('./rolo/data', entry.name, 'yolo_out/')
                if not os.path.exists(out_fold):
                    os.makedirs(out_fold)

                paths = load_folder(img_fold)
                gt_locations = load_dataset_gt(gt_file)

                for i, path in enumerate(paths):
                    filename = os.path.basename(path)
                    print("processing: ", i, ": ", filename)
                    img = file_to_img(path)

                    # Pass through YOLO layers
                    self.h_img, self.w_img, _ = img.shape

                    # todo, resize needed?

                    img_resized = cv2.resize(img, (448, 448))
                    img_RGB = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                    img_resized_np = np.asarray(img_RGB)
                    inputs = np.zeros((1, 448, 448, 3), dtype='float32')
                    inputs[0] = (img_resized_np / 255.0) * 2.0 - 1.0
                    in_dict = {self.x: inputs}

                    start_time = time.time()
                    feature = self.sess.run(self.fc_30, feed_dict=in_dict)
                    cycle_time = time.time() - start_time
                    print('cycle time= ', cycle_time)
                    total_time += cycle_time
                    output = self.sess.run(self.fc_32, feed_dict=in_dict)  # make sure it does not run conv layers twice

                    locations = self.interpret_output(output[0])
                    gt_location = find_gt_location(gt_locations, i)
                    location = self.find_best_location(locations,
                                                       gt_location)  # find the ROI that has the maximum IOU with the ground truth

                    debug_location(img, location, controller)
                    debug_gt_location(img, gt_location)

                    # change location into [0, 1]
                    loss = self.call_yolo_iou(location[1:5], gt_location)
                    location = location_from_0_to_1(self.w_img, self.h_img, location)
                    avg_loss += loss
                    total += 1
                    print("loss: ", loss)
                    yolo_output = np.concatenate(
                        (np.reshape(feature, [-1, self.num_feat]),
                         np.reshape(location, [-1, self.num_predict])),
                        axis=1
                    )
                    save_yolo_output(out_fold, yolo_output, filename)

        avg_loss = avg_loss / total
        print("YOLO avg_loss: ", avg_loss)

        print "Time Spent on Tracking: " + str(total_time)
        print "fps: " + str(i / total_time)

        return

    def build_networks(self):
        print("Building YOLO_small graph...")
        g_1 = tf.Graph()
        with g_1.as_default():
            self.x = tf.compat.v1.placeholder('float32', [None, 448, 448, 3])
            self.conv_1 = self.conv_layer(1, self.x, 64, 7, 2)
            self.pool_2 = pooling_layer(2, self.conv_1, 2, 2)
            self.conv_3 = self.conv_layer(3, self.pool_2, 192, 3, 1)
            self.pool_4 = pooling_layer(4, self.conv_3, 2, 2)
            self.conv_5 = self.conv_layer(5, self.pool_4, 128, 1, 1)
            self.conv_6 = self.conv_layer(6, self.conv_5, 256, 3, 1)
            self.conv_7 = self.conv_layer(7, self.conv_6, 256, 1, 1)
            self.conv_8 = self.conv_layer(8, self.conv_7, 512, 3, 1)
            self.pool_9 = pooling_layer(9, self.conv_8, 2, 2)
            self.conv_10 = self.conv_layer(10, self.pool_9, 256, 1, 1)
            self.conv_11 = self.conv_layer(11, self.conv_10, 512, 3, 1)
            self.conv_12 = self.conv_layer(12, self.conv_11, 256, 1, 1)
            self.conv_13 = self.conv_layer(13, self.conv_12, 512, 3, 1)
            self.conv_14 = self.conv_layer(14, self.conv_13, 256, 1, 1)
            self.conv_15 = self.conv_layer(15, self.conv_14, 512, 3, 1)
            self.conv_16 = self.conv_layer(16, self.conv_15, 256, 1, 1)
            self.conv_17 = self.conv_layer(17, self.conv_16, 512, 3, 1)
            self.conv_18 = self.conv_layer(18, self.conv_17, 512, 1, 1)
            self.conv_19 = self.conv_layer(19, self.conv_18, 1024, 3, 1)
            self.pool_20 = pooling_layer(20, self.conv_19, 2, 2)
            self.conv_21 = self.conv_layer(21, self.pool_20, 512, 1, 1)
            self.conv_22 = self.conv_layer(22, self.conv_21, 1024, 3, 1)
            self.conv_23 = self.conv_layer(23, self.conv_22, 512, 1, 1)
            self.conv_24 = self.conv_layer(24, self.conv_23, 1024, 3, 1)
            self.conv_25 = self.conv_layer(25, self.conv_24, 1024, 3, 1)
            self.conv_26 = self.conv_layer(26, self.conv_25, 1024, 3, 2)
            self.conv_27 = self.conv_layer(27, self.conv_26, 1024, 3, 1)
            self.conv_28 = self.conv_layer(28, self.conv_27, 1024, 3, 1)
            self.fc_29 = self.fc_layer(29, self.conv_28, 512, flat=True, linear=False)
            self.fc_30 = self.fc_layer(30, self.fc_29, 4096, flat=False, linear=False)
            # skip dropout_31
            self.fc_32 = self.fc_layer(32, self.fc_30, 1470, flat=False, linear=True)
            self.sess = tf.compat.v1.Session(graph=g_1)
            self.sess.run(tf.initialize_all_variables())
            self.saver = tf.compat.v1.train.Saver()
            self.saver.restore(self.sess, self.weights_file)
        print("Loading complete!" + '\n')

    def conv_layer(self, idx, inputs, filters, size, stride):
        channels = inputs.get_shape()[3]
        weight = tf.Variable(tf.random.truncated_normal([size, size, int(channels), filters], stddev=0.1))
        biases = tf.Variable(tf.constant(0.1, shape=[filters]))

        pad_size = size // 2
        pad_mat = np.array([[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])
        inputs_pad = tf.pad(inputs, pad_mat)

        conv = tf.nn.conv2d(inputs_pad, weight, strides=[1, stride, stride, 1], padding='VALID',
                            name=str(idx) + '_conv')
        conv_biased = tf.add(conv, biases, name=str(idx) + '_conv_biased')

        print(
                'Layer  %d : Type = Conv, Size = %d * %d, Stride = %d, Filters = %d, Input channels = %d' %
                (idx, size, size, stride, filters, int(channels))
        )

        return tf.maximum(self.alpha * conv_biased, conv_biased, name=str(idx) + '_leaky_relu')

    def fc_layer(self, idx, inputs, hiddens, flat=False, linear=False):
        input_shape = inputs.get_shape().as_list()

        if flat:
            dim = input_shape[1] * input_shape[2] * input_shape[3]
            inputs_transposed = tf.transpose(inputs, (0, 3, 1, 2))
            inputs_processed = tf.reshape(inputs_transposed, [-1, dim])
        else:
            dim = input_shape[1]
            inputs_processed = inputs

        weight = tf.Variable(tf.truncated_normal([dim, hiddens], stddev=0.1))
        biases = tf.Variable(tf.constant(0.1, shape=[hiddens]))

        print(
                'Layer  %d : Type = Full, Hidden = %d, Input dimension = %d, Flat = %d, Activation = %d' %
                (idx, hiddens, int(dim), int(flat), 1 - int(linear))
        )

        if linear:
            return tf.add(tf.matmul(inputs_processed, weight), biases, name=str(idx) + '_fc')
        ip = tf.add(tf.matmul(inputs_processed, weight), biases)

        return tf.maximum(self.alpha * ip, ip, name=str(idx) + '_fc')

    def interpret_output(self, output):
        probs = np.zeros((7, 7, 2, 20))
        class_probs = np.reshape(output[0:980], (7, 7, 20))
        scales = np.reshape(output[980:1078], (7, 7, 2))
        boxes = np.reshape(output[1078:], (7, 7, 2, 4))
        offset = np.transpose(np.reshape(np.array([np.arange(7)] * 14), (2, 7, 7)), (1, 2, 0))

        boxes[:, :, :, 0] += offset
        boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
        boxes[:, :, :, 0:2] = boxes[:, :, :, 0:2] / 7.0
        boxes[:, :, :, 2] = np.multiply(boxes[:, :, :, 2], boxes[:, :, :, 2])
        boxes[:, :, :, 3] = np.multiply(boxes[:, :, :, 3], boxes[:, :, :, 3])

        boxes[:, :, :, 0] *= self.w_img
        boxes[:, :, :, 1] *= self.h_img
        boxes[:, :, :, 2] *= self.w_img
        boxes[:, :, :, 3] *= self.h_img

        for i in range(2):
            for j in range(20):
                probs[:, :, i, j] = np.multiply(class_probs[:, :, j], scales[:, :, i])

        filter_mat_probs = np.array(probs >= self.threshold, dtype='bool')
        filter_mat_boxes = np.nonzero(filter_mat_probs)
        boxes_filtered = boxes[filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]
        probs_filtered = probs[filter_mat_probs]
        classes_num_filtered = np.argmax(filter_mat_probs, axis=3)[
            filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]

        argsort = np.array(np.argsort(probs_filtered))[::-1]
        boxes_filtered = boxes_filtered[argsort]
        probs_filtered = probs_filtered[argsort]
        classes_num_filtered = classes_num_filtered[argsort]

        for i in range(len(boxes_filtered)):
            if probs_filtered[i] == 0: continue
            for j in range(i + 1, len(boxes_filtered)):
                if iou(boxes_filtered[i], boxes_filtered[j]) > self.iou_threshold:
                    probs_filtered[j] = 0.0

        filter_iou = np.array(probs_filtered > 0.0, dtype='bool')
        boxes_filtered = boxes_filtered[filter_iou]
        probs_filtered = probs_filtered[filter_iou]
        classes_num_filtered = classes_num_filtered[filter_iou]

        result = []
        for i in range(len(boxes_filtered)):
            result.append([self.classes[classes_num_filtered[i]], boxes_filtered[i][0], boxes_filtered[i][1],
                           boxes_filtered[i][2], boxes_filtered[i][3], probs_filtered[i]])

        return result

    def find_best_location(self, locations, gt_location):
        # locations (class, x, y, w, h, prob); (x, y) is the middle pt of the rect
        # gt_location (x1, y1, w, h)
        x1 = gt_location[0]
        y1 = gt_location[1]
        w = gt_location[2]
        h = gt_location[3]
        gt_location_revised = [x1 + w / 2, y1 + h / 2, w, h]

        max_ious = 0
        for id, location in enumerate(locations):
            location_revised = location[1:5]
            print("location: ", location_revised)
            print("gt_location: ", gt_location_revised)
            ious = iou(location_revised, gt_location_revised)
            if ious >= max_ious:
                max_ious = ious
                index = id
        print("Max IOU: " + str(max_ious))
        if max_ious != 0:
            best_location = locations[index]
            class_index = self.classes.index(best_location[0])
            best_location[0] = class_index
            return best_location
        else:  # it means the detection failed, no intersection with the ground truth
            return [0, 0, 0, 0, 0, 0]

    def call_yolo_iou(self, location, gt_location):
        # Translate yolo's box mid-point (x0, y0) to top-left point (x1, y1), in order to compare with gt
        location[0] = location[0] - location[2] / 2
        location[1] = location[1] - location[3] / 2
        loss = iou(location, gt_location)
        return loss

    def predict_location_from_image(self, image_path):
        img = file_to_img(image_path)

        # Pass through YOLO layers
        self.h_img, self.w_img, _ = img.shape

        # todo, resize needed?
        # todo, change tensor placeholder to 500, 500 image
        img_resized = cv2.resize(img, (448, 448))
        img_RGB = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_resized_np = np.asarray(img_RGB)

        # todo, get image dim from single place
        inputs = np.zeros((1, 448, 448, 3), dtype='float32')
        inputs[0] = (img_resized_np / 255.0) * 2.0 - 1.0
        in_dict = {self.x: inputs}

        feature = self.sess.run(self.fc_30, feed_dict=in_dict)
        output = self.sess.run(self.fc_32, feed_dict=in_dict)
        if len(output) > 0:
            locations = self.interpret_output(output[0])
            if len(locations) > 0:
                return locations[0][1:5]
            else:
                return None
        else:
            return None
