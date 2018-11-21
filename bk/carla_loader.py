from random import shuffle

import numpy as np
import h5py
import tensorflow as tf


class CarlaDataLoader(object):
    def __init__(self,
                 sess,
                 data_folder,
                 data_list,
                 eval_folder,
                 eval_list,
                 train_batch_size,
                 eval_batch_size,
                 sequnece_len,
                 shuffle_list=True):

        self.sess = sess
        self.sequnece_len = sequnece_len

        # training parameters
        self.data_folder = data_folder
        self.data_list = data_list
        self.full_len = list(range(len(data_list) * sequnece_len))
        if shuffle_list:
            shuffle(self.full_len)
        self.idx = 0

        # test parameters
        self.eval_folder = eval_folder
        self.eval_list = eval_list

        self.train_dataset, self.train_iter, self.train_handle = \
            self.build_dataset(self.train_gen,
                               self.train_parse,
                               train_batch_size)

        self.eval_dataset, self.eval_iter, self.eval_handle = \
            self.build_dataset(self.eval_gen,
                               self.eval_parse,
                               eval_batch_size)

        self.handle = tf.placeholder(tf.string, shape=[])
        self.iterator = tf.data.Iterator.from_string_handle(
            self.handle,
            self.train_dataset.output_types,
            self.train_dataset.output_shapes)
        self.next = self.iterator.get_next()

    def build_dataset(self, gen, parse_fn, batch_size):
        # TODO remove hardcode parameters of buffer
        dataset = tf.data.Dataset.from_generator(
            generator=gen,
            output_types=(tf.float32, tf.float32),
            output_shapes=(tf.TensorShape([88, 200, 3]), tf.TensorShape([28])))

        # dataset = dataset.batch(batch_size=batch_size)
        dataset = dataset.apply(tf.data.experimental.map_and_batch(
            map_func=parse_fn, batch_size=batch_size))
        dataset = dataset.prefetch(buffer_size=3)
        data_iter = dataset.make_initializable_iterator()
        data_handle = self.sess.run(data_iter.string_handle())
        return dataset, data_iter, data_handle

    def train_gen(self):
        for i in self.full_len:
            data_idx = self.full_len[i] // self.sequnece_len
            file_idx = self.full_len[i] % self.sequnece_len
            file_name = '%s/%s' % (self.data_folder, self.data_list[data_idx])

            with h5py.File(file_name, 'r') as h5_file:
                img = np.array(h5_file['rgb'])[file_idx]
                target = np.array(h5_file['targets'])[file_idx]
                # img = tf.convert_to_tensor(
                #     np.array(h5_file['rgb'])[file_idx],
                #     dtype=tf.float32)
                # target = tf.convert_to_tensor(
                #     np.array(h5_file['targets'])[file_idx],
                #     dtype=tf.float32)
            yield (img, target)

    def train_parse(self, img, target):
        # from numpy to tensorflow
        # data augmentation
        return (img, target)

    def eval_gen(self):
        for item in self.eval_list:
            for i in range(self.sequnece_len):
                file_name = '%s/%s' % (
                    self.eval_folder,
                    item)

                h5_file = h5py.File(file_name, 'r')
                img = tf.conver_to_tensor(
                    np.array(h5_file['rgb'])[i],
                    dtype=tf.float32)
                target = tf.convert_to_tensor(
                    np.array(h5_file['targets'])[i],
                    dtype=tf.float32)
                yield (img, target)

    def eval_parse(self, img, target):
        # from numpy to tensorflow
        return img, target
