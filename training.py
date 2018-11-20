#!/usr/bin/env python
# coding=utf-8

'''
Author:Tai Lei
Date:Thursday, March 08, 2018 PM08:38:54 HKT
Info:
'''

import tensorflow as tf
from PIL import Image as PilImage
import numpy as np

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist
from deepdrive_msgs.msg import Control
from imitation_learning_network import load_imitation_learning_network

import rospy, rospkg
import os, sys
import cv2

def crop_height(img_, crop_margin_):
    return img_[crop_margin_:crop_margin_*-1, :]

def crop_width(img_, crop_margin_):
    return img_[:, crop_margin_:crop_margin_*-1]

def crop_none(img_, crop_margin_):
    return img_


class CarlaPerception():

    def __init__(self, topic_name, crop_size):
        config_gpu = tf.ConfigProto()
        config_gpu.gpu_options.visible_device_list = '0'
        config_gpu.gpu_options.per_process_gpu_memory_fraction = 0.25

        #self.dropout_vec = [1.0] * 8 + [0.7] * 2 + [0.5] * 2 + [0.5] * 1 + [0.5, 1.] * 5
        self.dropout_vec = [1.0] * 8 + [1.0] * 2 + [1.0] * 2 + [1.0] * 1 + [1.0, 1.0] * 5

        self._sess = tf.Session(config=config_gpu)

        self._image_size = (88, 200, 3)
        self.crop_size = crop_size
        self._avoid_stopping =  False
        rpk = rospkg.RosPack()
        dir_path = rpk.get_path("carla_perception")
        self._models_path = dir_path + '/src/model/'

        with tf.device('/gpu:0'):
            self._input_images = tf.placeholder(
                    "float", shape=[None,
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

        self._sess.run(tf.global_variables_initializer())
        self.load_model()

        self.pred_pub = rospy.Publisher('contol_prediction', Control, queue_size=1)
        self.image_pub = rospy.Publisher('output_image', Image, queue_size=1)
        self.image_sub = rospy.Subscriber(topic_name,
                Image,
                self.callback,
                queue_size=1)

        self.control_cmd = Control()
        self.cv2ros_bridge = CvBridge()

        self.joy_sub = rospy.Subscriber("joy_teleop/joy",
                Joy, self.joy_callback, queue_size=1)
        self.twist_pub = rospy.Publisher("cmd_vel", Twist,
                queue_size=1)
        self.direction=2.0
        self.direction_text = "Follow"
        self.cil_control_flag=False
        self.linear_vel = 0.0
        self.angle_scale = float(rospy.get_param("cil_angle_scale"))
        self.linear_scale = float(rospy.get_param("cil_linear_scale"))
        self.control_twist = Twist()
        print("initialization over")

    def load_model(self):
        variables_to_restore = tf.global_variables()
        saver = tf.train.Saver(variables_to_restore, max_to_keep=0)
        print ("models_path====================", self._models_path)
        if not os.path.exists(self._models_path):
            raise RuntimeError('failed to find the models path')
        ckpt = tf.train.get_checkpoint_state(self._models_path)
        if ckpt:
            print ('Restoring from ', ckpt.model_checkpoint_path)
            saver.restore(self._sess, ckpt.model_checkpoint_path)
        else:
            ckpt = 0
        return ckpt

    #def run_step(self, img, speed, target):
        #control = self._compute_action(img, speed, direction)

    def _compute_action(self, image_input, speed, direction=None):

        #rgb_image = rgb_image[self._image_cut[0]:self._image_cut[1], :]
        #rgb_image = rgb_image[self._image_cut[0]:self._image_cut[1], :]
        #print (rgb_image.shape)
        #rgb_image = rgb_image[:960-465, :]


        image_input = image_input.astype(np.float32)
        image_input = np.multiply(image_input, 1.0 / 255.0)

        steer, acc, brake = self._control_function(image_input, speed, direction, self._sess)

        # This a bit biased, but is to avoid fake breaking

        #if brake < 0.1:
            #brake = 0.0

        #if acc > brake:
            #brake = 0.0

        # We limit speed to 35 km/h to avoid
        #if speed > 35.0 and brake == 0.0:
            #acc = 0.0

        #control = Control()
        #control.steer = steer
        #control.throttle = acc
        #control.brake = brake

        #control.hand_brake = 0
        #control.reverse = 0

        return steer, acc, brake

    def _control_function(self, image_input, speed, control_input, sess):

        branches = self._network_tensor
        x = self._input_images
        dout = self._dout
        input_speed = self._input_data[1]

        image_input = image_input.reshape((1, self._image_size[0], self._image_size[1], self._image_size[2]))

        # Normalize with the maximum speed from the training set ( 90 km/h)
        speed = np.array(speed / 90.0)

        speed = speed.reshape((1, 1))

        if control_input == 2 or control_input == 0.0:
            all_net = branches[0]
        elif control_input == 3:
            all_net = branches[2]
        elif control_input == 4:
            all_net = branches[3]
        else:
            all_net = branches[1]

        feedDict = {x: image_input, input_speed: speed, dout: [1] * len(self.dropout_vec)}

        output_all = sess.run(all_net, feed_dict=feedDict)

        predicted_steers = (output_all[0][0])

        predicted_acc = (output_all[0][1])

        predicted_brake = (output_all[0][2])

        if self._avoid_stopping:
            predicted_speed = sess.run(branches[4], feed_dict=feedDict)
            predicted_speed = predicted_speed[0][0]
            real_speed = speed * 90.0

            real_predicted = predicted_speed * 90.0
            if real_speed < 5.0 and real_predicted > 6.0:

                predicted_acc = 1 * (20.0 / 90.0 - speed) + predicted_acc

                predicted_brake = 0.0

                predicted_acc = predicted_acc[0][0]


        return predicted_steers, predicted_acc, predicted_brake

    def callback(self, msg):
        test_file = self.cv2ros_bridge.imgmsg_to_cv2(msg, 'rgb8')
        if self.crop_size != None:
            test_file = test_file[-1*self.crop_size:, :, :]
        image_input = cv2.resize(test_file, (self._image_size[1],self._image_size[0]))

        speed = 25

        steer,accel,brake =  self._compute_action(image_input, speed, self.direction)

        cv2.rectangle(image_input,
                (int(0.5*self._image_size[1]), self._image_size[0]-20),
                (int(0.5*self._image_size[1]+int(steer*50)), self._image_size[0]-10),
                (255,0,0),
                thickness=-1)
        cv2.rectangle(image_input,
                (165, 40),
                (175, 40-int(accel*10)),
                (0,255,255),
                thickness=-1)
        cv2.rectangle(image_input,
                (180, 40),
                (190, 40-int(brake*10)),
                (0,0,255),
                thickness=-1)
        cv2.putText(image_input, self.direction_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0))
        output_msg = self.cv2ros_bridge.cv2_to_imgmsg(image_input, 'rgb8')

        self.control_cmd.steerCmd = steer
        self.control_cmd.accelCmd = accel
        self.control_cmd.brakeCmd = brake
        self.pred_pub.publish(self.control_cmd)
        self.image_pub.publish(output_msg)
        if self.cil_control_flag==True:
            self.control_twist.angular.z = steer*self.angle_scale*-1
            self.control_twist.linear.x = self.linear_vel*self.linear_scale
            self.twist_pub.publish(self.control_twist)

    def joy_callback(self, msg):
        if msg.buttons[3]==1:
            self.direction=3.0
            self.direction_text = "LEFT"
        elif msg.buttons[1]==1:
            self.direction=4.0
            self.direction_text = "RIGHT"
        elif msg.buttons[4]==1:
            self.direction=5.0
            self.direction_text = "STRAIGHT"
        else:
            self.direction=2.0
            self.direction_text = "FOLLOW"

        if msg.buttons[5]==1:
            self.linear_vel = msg.axes[1]
            self.cil_control_flag=True
            linear_scale = float(rospy.get_param("cil_linear_scale"))
            self.control_twist.linear.x = self.linear_vel*linear_scale
            self.twist_pub.publish(self.control_twist)
        else:
            self.cil_control_flag=False

if __name__=="__main__":

    # for bulldog raw data

    enable_vr = rospy.get_param('enable_vr')
    if enable_vr==True:
        topic_name_ = '/transfer_image'
        #load_size = (640, 400)
        crop_size = 282
    else:
        topic_name_ = '/camera/image_color'
        #load_size = (1920, 1200)
        crop_size = 845
    obj = CarlaPerception(topic_name_, crop_size)
    rospy.init_node('deepdrive_perception', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
