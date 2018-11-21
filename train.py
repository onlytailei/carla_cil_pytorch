#!/usr/bin/env python
# coding=utf-8

'''
Author:Tai Lei
Date:Thursday, March 08, 2018 PM08:38:54 HKT
Info:
'''

import os
import cv2
import tensorflow as tf
import numpy as np

from imitation_learning_network import load_imitation_learning_network
from carla_loader import CarlaDataLoader


class CarlaTraining():

    def __init__(self):
        config_gpu = tf.ConfigProto()
        config_gpu.gpu_options.visible_device_list = '0'
        config_gpu.gpu_options.per_process_gpu_memory_fraction = 0.25

        self.dropout_vec = [1.0] * 8 + [0.7] * 2 + [0.5] * 2 \
            + [0.5] * 1 + [0.5, 1.] * 5
        # self.dropout_vec = [1.0] * 8 + [1.0] * 2 + [1.0] * 2 \
        #     + [1.0] * 1 + [1.0, 1.0] * 5

        self.target_idx = 10

        self.sess = tf.Session(config=config_gpu)
        self._image_size = (88, 200, 3)

        with tf.device('/gpu:0'):
            self._input_images = tf.placeholder(
                    "float", shape=[
                        None,
                        self._image_size[0],
                        self._image_size[1],
                        self._image_size[2]],
                    name="input_image")
            self._input_data = []
            self._input_data.append(tf.placeholder(tf.float32,
                                                   shape=[None, 4],
                                                   name="input_control"))

            self._input_data.append(tf.placeholder(tf.float32,
                                                   shape=[None, 1],
                                                   name="input_speed"))
            self._dout = tf.placeholder("float", shape=[len(self.dropout_vec)])

        with tf.name_scope("Network"):
            self._network_tensor = load_imitation_learning_network(
                    self._input_images,
                    self._input_data,
                    self._image_size,
                    self._dout)

        self.sess.run(tf.global_variables_initializer())
        # self.load_model()

        train_folder = "/home/tai/ws/ijrr_2018/carla_cil_dataset/AgentHuman/SeqTrain"
        test_folder = "/home/tai/ws/ijrr_2018/carla_cil_dataset/AgentHuman/SeqTrain"

        with open("../carla_cil_dataset/AgentHuman/test_loader.txt", "r") as f:
            name_list = f.readlines()

        name_list = [item.split()[0] for item in name_list]

        self.loader = CarlaDataLoader(
            sess, train_folder, name_list, eval_folder, name_list,
            train_batch_size=5, eval_batch_size=5, sequnece_len=200)

        print("initialization over")

    def train_loop(self):
        epoch = 0
        while epoch < 20:  # TODO hard code
            self.sess.run(self.loader.train_iter.initializer)
            while True:  # train loop
                try:
                    img, target = self.sess.run(
                        self.loader.next,
                        feed_dict={self.loader.handle:
                                   self.loader.train_handle})
                    self.train_func()
                except tf.errors.OutOfRangeError:
                    epoch += 1
                    break

            self.sess.run(self.loader.eval_iter.initializer)
            while True:  # eval loop
                try:
                    img, target = self.sess.run(
                        self.loader.next,
                        feed_dict={self.loader.handle:
                                   self.loader.eval_handle})
                except tf.errors.OutOfRangeError:
                    break

    def train_func(self, img, target):
        branches = self._network_tensor
        x = self._input_images
        dout = self._dout
        input_speed = self._input_data[1]

        speed = np.array(target[:, self.speed_idx] / 90.0)

        if control_input == 2 or control_input == 0.0:
            all_net = branches[0]
        elif control_input == 3:
            all_net = branches[2]
        elif control_input == 4:
            all_net = branches[3]
        else:
            all_net = branches[1]

        feedDict = {x: image_input,
                    input_speed: speed,
                    dout: [1] * len(self.dropout_vec)}

        output_all = sess.run(all_net, feed_dict=feedDict)

        predicted_steers = (output_all[0][0])
        predicted_acc = (output_all[0][1])
        predicted_brake = (output_all[0][2])

        return predicted_steers, predicted_acc, predicted_brake

    def load_model(self):
        variables_to_restore = tf.global_variables()
        saver = tf.train.Saver(variables_to_restore, max_to_keep=0)
        print("models_path====================", self._models_path)
        if not os.path.exists(self._models_path):
            raise RuntimeError('failed to find the models path')
        ckpt = tf.train.get_checkpoint_state(self._models_path)
        if ckpt:
            print('Restoring from ', ckpt.model_checkpoint_path)
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            ckpt = 0
        return ckpt


if __name__ == "__main__":
    pass
