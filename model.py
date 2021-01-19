import framework as fm
from DS import DS
from tensorflow.compat import v1 as tf
from matplotlib import pyplot as plt
import utils.CommonMetrics as cm
import numpy as np
import pandas as pd
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class MyTensors(fm.Tensors):
    def compute_grads(self, opt):
        vars = tf.trainable_variables()
        grads = []
        for gpu_id, ts in enumerate(self.sub_ts):
            with tf.device('/gpu:%d' % gpu_id):
                grads.append([opt.compute_gradients(ts.losses[0], vars)])
        return [self.get_grads_mean(grads, i) for i in range(len(grads[0]))]


class Config(fm.Config):

    def __init__(self):
        super().__init__()
        self.frame = 40
        self.hidden_size = 1024
        self.input_size = 128 * 3
        self.f_size = 256
        self.z_size = 32
        self.z_hidden_size = 128
        self.training = True
        self.ds = None
        self.lr = 0.0001
        self.epoches = 500
        self.batch_size = 20

    def get_tensors(self):
        return MyTensors(self)

    def get_sub_tensors(self, gpu_index):
        return SubTensor(self)

    def get_name(self):
        return "test11"

    def get_app(self):
        return App(self)

    def get_ds_train(self):
        self.read_ds()
        return self.ds

    def get_ds_test(self):
        self.read_ds()
        return self.ds

    def test(self):
        self.training = False
        super(Config, self).test()

    def read_ds(self):
        if self.ds is None:
            self.ds = DS(self)


class SubTensor:
    def __init__(self, config: Config):
        self.config = config
        self.x = tf.placeholder(dtype=tf.float64, shape=[None, config.frame, config.input_size], name="x")  # [-1, 40, 8064]
        self.inputs = [self.x]
        x = tf.reshape(self.x, [-1, config.input_size])  # [-1, 40, 8064]

        vec = self.encoder_frame(x, "encoder_frame")  # [-1, 40, 1024]

        self.mean_f, self.logvar_f = self.encoder_f(vec, "encoder_f")  # [-1, 256]
        self.f = self.reparameterize(self.mean_f, self.logvar_f)  # [-1, 256]
        f = tf.reshape(self.f, [-1, 1, self.f.shape[-1]])  # [-1, 1, 256]
        f = tf.tile(f, [1, config.frame, 1])  # [-1, 40, 256]

        self.mean_z, self.logvar_z = self.encoder_z(vec, f, "encoder_z")  # [-1, 40, 32]
        self.z = self.reparameterize(self.mean_z, self.logvar_z)  # [-1, 40, 32]

        zf = tf.concat((self.z, f), axis=2)  # [-1, 40, 256 + 32]
        self.predict_y = self.decoder_frame(zf, "decoder_frame")  # [-1, 40, 8064]

        self.loss()

    def loss(self):
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.predict_y - self.x), axis=(1, 2)))

        kld_f = tf.reduce_mean(
            -0.5 * tf.reduce_sum(1 + self.logvar_f - tf.pow(self.mean_f, 2) - tf.exp(self.logvar_f), 1))
        self.losses = [tf.sqrt(loss) + kld_f]

    def encoder_frame(self, x, name):
        """
        :param x: [-1, 8064]
        :return:  [-1, 40, 2048]
        """
        with tf.variable_scope(name):
            x = tf.reshape(x, [-1, self.config.input_size, 1])  # [-1, 8064, 1]

            ef_1 = tf.layers.conv1d(x, 16, 3, 3, padding="same", activation=tf.nn.relu, name="conv_1")  # [-1, 2688, 16]
            ef_2 = tf.layers.conv1d(ef_1, 32, 3, 3, padding="same", activation=tf.nn.relu, name="conv_2")  # [-1, 896, 32]

            flatten = tf.layers.flatten(ef_2)  # [-1, 32 * 896]
            vec = tf.layers.dense(flatten, self.config.hidden_size, activation=tf.nn.relu, name="dense_1")  # [-1, 1024]

            vec = tf.reshape(vec, [-1, self.config.frame, self.config.hidden_size])
        return vec

    def encoder_f(self, x, name):
        """
        :param x: [-1, 40, 1024]
        :param name:
        :return: [-1, 256]
        """
        with tf.variable_scope(name):
            x = tf.layers.dense(x, self.config.f_size, name="dense1")  # [-1, 40, 256]

            cell_l2r = tf.nn.rnn_cell.LSTMCell(self.config.f_size, name="cell_l2r", state_is_tuple=False)
            cell_r2l = tf.nn.rnn_cell.LSTMCell(self.config.f_size, name="cell_r2l", state_is_tuple=False)
            batch_size = tf.shape(x)[0]
            state_l2r = cell_l2r.zero_state(batch_size, dtype=x.dtype)
            state_r2l = cell_r2l.zero_state(batch_size, dtype=x.dtype)
            for i in range(self.config.frame):
                y_l2r, state_l2r = cell_l2r(x[:, i, :], state_l2r)  # y_l2r : [-1, 256]
                y_r2l, state_r2l = cell_r2l(x[:, self.config.frame - i - 1, :], state_r2l)  # y_r2l : [-1, 256]

            y = tf.concat((y_l2r, y_r2l), axis=1)  # [-1, 512]
            mean_f = tf.layers.dense(y, self.config.f_size, name="dense_mean")  # [-1, 256]
            logvar_f = tf.layers.dense(y, self.config.f_size, name="dense_logvar")  # [-1, 256]
        return mean_f, logvar_f

    def encoder_z(self, vec, f, name):
        """ 
        :param vec: [-1, 40, 1024]
        :param f: [-1, 40, 256]
        :param name: 
        :return: [-1, 40, 32]
        """""
        with tf.variable_scope(name):
            x = tf.concat((vec, f), axis=2)  # [-1, 40, 1024 + 256]
            x = tf.layers.dense(x, self.config.z_hidden_size, name="dense_1")  # [-1, 40, 128]

            cell_l2r = tf.nn.rnn_cell.LSTMCell(self.config.z_hidden_size, name="cell_l2r", state_is_tuple=False)
            cell_r2l = tf.nn.rnn_cell.LSTMCell(self.config.z_hidden_size, name="cell_r2l", state_is_tuple=False)
            batch_size = tf.shape(x)[0]
            state_l2r = cell_l2r.zero_state(batch_size, dtype=x.dtype)
            state_r2l = cell_r2l.zero_state(batch_size, dtype=x.dtype)
            y_l2r = []
            y_r2l = []  # [40, -1, 128]
            for i in range(self.config.frame):  # 40
                yi_l2r, state_l2r = cell_l2r(x[:, i, :], state_l2r)  # y_l2r : [-1, 128]
                yi_r2l, state_r2l = cell_r2l(x[:, self.config.frame - i - 1, :], state_r2l)  # y_r2l : [-1, 128]
                y_l2r.append(yi_l2r)
                y_r2l.insert(0, yi_r2l)
            y_lstm = [yi_l2r + yi_r2l for yi_l2r, yi_r2l in zip(y_l2r, y_r2l)]  # [40, -1, 128]
            y_lstm = tf.transpose(y_lstm, [1, 0, 2])  # [-1, 40, 128]

            rnn_cell = tf.nn.rnn_cell.BasicRNNCell(self.config.z_hidden_size * 2)
            # keras.layers.RNN(cell)
            features, states = tf.nn.dynamic_rnn(rnn_cell, y_lstm, dtype=tf.float64)  # [-1, 40, 256]

            mean_z = tf.layers.dense(features, self.config.z_size, name="dense_mean")  # [-1, 40, 32]
            logvar_z = tf.layers.dense(features, self.config.z_size, name="dense_logvar")  # [-1, 40, 32]
        return mean_z, logvar_z

    def decoder_frame(self, zf, name):
        """
        :param zf: [-1, 40, 512 + 32]
        :return:  [-1, 40, 8064]
        """""
        with tf.variable_scope(name):
            x = tf.reshape(zf, [-1, self.config.f_size + self.config.z_size])  # [-1, 256 + 32]

            df_1 = tf.layers.dense(x, self.config.hidden_size, activation=tf.nn.relu, name="dense_1")  # [-1, 1024]

            df_2 = tf.layers.dense(df_1, self.config.input_size, name="dense_2")  # [-1, 8064]
            y = tf.reshape(df_2, [-1, self.config.frame, self.config.input_size])  # [-1, 40, 8064]
        return y

    def reparameterize(self, mean, logvar, random_sampling=True):
        if random_sampling is True:
            std = tf.exp(0.5 * logvar)
            eps = tf.random.normal(shape=tf.shape(logvar), dtype=logvar.dtype)
            return mean + std * eps
        else:
            return mean + logvar * 0


class App(fm.App):
    def test(self, ds_test):
        ts = self.ts.sub_ts[0]
        data = ds_test.next_batch(1)[0]  # [1, 40, 8064]
        y = self.session.run(ts.predict_y, {ts.x: data})  # [1, 40, 8064]
        # todo
        x = [i for i in range(self.config.input_size)]

        # for i in range(40):
        #     plt.subplot(2, 40, i + 1)
        #     plt.plot(x, data[0][i])
        #
        # for i in range(40):
        #     plt.subplot(40, 2, i + 41)
        #     plt.plot(x, y[0][i])

        print("ED = {ed}".format(ed=cm.ed(data[0][0], y[0][0])))
        print("PCCD = {pccd}".format(pccd=cm.pccd(data[0][0], y[0][0])))
        print("SKLD = {skld}".format(skld=cm.skld(data[0][0], y[0][0])))
        print("HD = {hd}".format(hd=cm.hd(data[0][0], y[0][0])))
        # plt.subplot(2, 1, 1)
        plt.plot(x, data[0][0])
        # # # plt.subplot(2, 1, 2)
        plt.plot(x, y[0][0])
        plt.show()


# def test01(data):
#     x = [i + 1 for i in range(len(data))]
#     p = []
#     for i in range(len(data)):
#         pccds = []
#         for j in range(len(data)):
#             pccd = np.around(cm.pccd(data[i], data[j]), 4)
#             pccds.append(pccd)
#         p.append(pccds)
#     df = pd.DataFrame(p, columns=x)
#     print(df)


if __name__ == '__main__':
    config = Config()
    # x = [i for i in range(config.input_size)]
    # data = config.get_ds_train().next_batch(1)[0][0]  # [40, 8064]
    # test01(data)
    # plt.table(cellText=p, rowLabels=x, colLabels=x)
    # plt.show()

    # for i in range(5):
    #     plt.subplot(5, 1, i + 1)
    #     plt.plot(x, data[0][i])
    # plt.show()
    # config.from_cmd()
    # config.call("test")
    config.get_ds_train()