# from DS_2 import DS
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from DS_For_F5 import Data as DS
embedding = keras.layers.Embedding(7, 2, input_length=5)


def math_pccd(x, y):
    """
    x: [8, 128, 22]
    y: [8, 128, 22]
    pearson相关系数(皮尔逊相关系数) : [-1, 1]之间，0代表无相关，-1代表负相关，1代表正相关
    :return [0, 1], 返回值越大越相似
    """
    mean_x = np.mean(x, axis=1)[:, np.newaxis, :]
    mean_y = np.mean(y, axis=1)[:, np.newaxis, :]

    d = np.sum((x - mean_x) * (y - mean_y), axis=1) / (np.sqrt(np.sum(np.square(x - mean_x), axis=1)) * np.sqrt(np.sum(np.square(y - mean_y), axis=1)))
    d = np.mean(d)
    s = np.abs(d)  # 取绝对值，返回值在[0, 1]区间
    return s


def load_model(path):
    try:
        decoder_model = tf.keras.models.load_model(path)
        if decoder_model is None:
            assert "no model to loading!"
        return decoder_model
    except:
        assert "no model to loading!"


def reparameterize_tf(mean, logvar, random_sampling=True):
    if random_sampling is True:
        std = tf.exp(0.5 * logvar)
        eps = tf.random.normal(shape=tf.shape(logvar), dtype=logvar.dtype)
        return mean + std * eps
    else:
        return mean + logvar * 0


class Config:
    def __init__(self):
        self.model_path = "models/{name}/vae_model.h5".format(name=self.get_name())
        self.encoder_frame_path = "models/{name}/encoder_frame.h5".format(name=self.get_name())
        self.encoder_f_path = "models/{name}/encoder_f.h5".format(name=self.get_name())
        self.encoder_z_path = "models/{name}/encoder_z.h5".format(name=self.get_name())
        self.simple_z_path = "models/{name}/simple_z.h5".format(name=self.get_name())
        self.decoder_frame_path = "models/{name}/decoder_frame.h5".format(name=self.get_name())
        self.log_dir = "logs/{name}".format(name=self.get_name())
        self.training = True
        self.ds = None
        self.lr = 0.0001
        self.epoch = 50
        self.batch_size = 20
        # inputs shape
        self.frame = 8
        self.input_size = 128
        self.channel = 22
        # vae param
        self.hidden_size = 1024
        # f param
        self.f_size = 256
        # z param
        self.z_size = 16
        self.z_hidden_size = 128

    def get_name(self):
        return "test15"

    def get_ds_train(self):
        self.read_ds()
        return self.ds.simple()

    def get_ds_test(self):
        self.read_ds()
        return self.ds.simple()

    def test(self):
        self.training = False

    def read_ds(self):
        if self.ds is None:
            self.ds = DS(self)


class Encoder_Frame:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.inputs = None
        self.model = None

    def init_net(self):
        cfg = self.cfg
        activation = tf.nn.elu
        self.inputs = keras.Input(shape=[cfg.frame, cfg.input_size, cfg.channel], dtype=tf.float32)  # [None, 8, 128, 40]
        # attention
        x = self.attention()
        x = tf.reshape(x, [-1, cfg.input_size, cfg.channel])  # [None, 128, 40]
        base_filters = 512
        for i in range(4):
            # [-1, 128, 40] -> [-1, 64, 512] -> [-1, 32, 1024] -> [-1, 16, 2048] -> [-1, 8, 4096]
            x = keras.layers.Conv1D(filters=base_filters,
                                    kernel_size=3,
                                    strides=2,
                                    padding='same',
                                    activation=activation)(x)
            base_filters *= 2
        encoded_signal = keras.layers.Flatten()(x)  # 8 * 1024
        vec = keras.layers.Dense(cfg.hidden_size)(encoded_signal)  # [-1, 1024]
        vec = tf.reshape(vec, [-1, cfg.frame, cfg.hidden_size])  # [-1, 8, 1024]
        self.model = keras.Model(inputs=self.inputs, outputs=vec)

    def attention(self):
        # 假设时间之间存在某种关系
        cfg = self.cfg
        q = keras.layers.Dense(units=cfg.channel, use_bias=False)(self.inputs)  # [-1, 8, 128, 40]
        k = keras.layers.Dense(units=cfg.channel, use_bias=False)(self.inputs)  # [-1, 8, 128, 40]
        v = keras.layers.Dense(units=cfg.channel, use_bias=False)(self.inputs)  # [-1, 8, 128, 40]
        score = tf.matmul(q, tf.transpose(k, [0, 1, 3, 2]))  # [-1, 8, 128, 128]
        score = tf.nn.softmax(score)  # [-1, 8, 128, 128]
        x = tf.matmul(score, v)  # [-1, 8, 128, 40]
        return x

    def __call__(self, x):
        return self.model(x)

    def get_inputs(self):
        return self.inputs

    def save(self):
        self.model.save(filepath=self.cfg.encoder_frame_path)

    def load(self):
        if self.model is None:
            self.model = load_model(self.cfg.encoder_frame_path)

    def predict(self, inputs):
        return self.model.predict(inputs)


class Encoder_F:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.inputs = None
        self.model = None

    def init_net(self):
        cfg = self.cfg
        inputs = keras.Input(shape=[cfg.frame, cfg.hidden_size], dtype=tf.float32)  # [-1, 8, 1028]
        forward = tf.keras.layers.LSTM(cfg.f_size, return_sequences=False, go_backwards=False)  # [-1, 256]
        backward = tf.keras.layers.LSTM(cfg.f_size, return_sequences=False, go_backwards=True)  # [-1, 256]
        y = keras.layers.Bidirectional(layer=forward, backward_layer=backward, merge_mode='concat')(inputs)  # [-1, 512]
        mean_f = keras.layers.Dense(cfg.f_size)(y)  # [-1, 256]
        log_var_f = keras.layers.Dense(cfg.f_size)(y)  # [-1, 256]
        self.model = keras.Model(inputs=inputs, outputs=[mean_f, log_var_f])

    def __call__(self, x):
        return self.model(x)

    def get_inputs(self):
        return self.inputs

    def save(self):
        self.model.save(filepath=self.cfg.encoder_f_path)

    def load(self):
        if self.model is None:
            self.model = load_model(self.cfg.encoder_f_path)

    def predict(self, inputs):
        return self.model.predict(inputs)


class Encoder_Z:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.inputs = None
        self.model = None

    def init_net(self):
        cfg = self.cfg
        vec = keras.Input(shape=[cfg.frame, cfg.hidden_size], dtype=tf.float32)  # [-1, 8, 1024]
        f = keras.Input(shape=[cfg.f_size], dtype=tf.float32)  # [-1, 256]
        new_f = keras.layers.Reshape([1, cfg.f_size])(f)  # [-1, 1, 256]
        new_f = tf.tile(new_f, [1, config.frame, 1])  # [-1, 8, 256]
        x = tf.concat((vec, new_f), axis=2)  # [-1, 8, 1024 + 256]
        forward = keras.layers.LSTM(cfg.z_hidden_size, return_sequences=True)  # [-1, 8, 128]
        backward = keras.layers.LSTM(cfg.z_hidden_size, return_sequences=True, go_backwards=True)  # [-1, 8, 128]
        lstm = keras.layers.Bidirectional(layer=forward, backward_layer=backward, merge_mode="concat")(x)  # [-1, 8, 256]
        rnn = keras.layers.SimpleRNN(units=cfg.z_hidden_size, return_sequences=True)(lstm)  # [-1, 8, 128]
        mean_z = keras.layers.Dense(cfg.z_size)(rnn)  # [-1, 8, 32]
        log_var_z = keras.layers.Dense(cfg.z_size)(rnn)  # [-1, 8, 32]
        self.model = keras.Model(inputs=[vec, f], outputs=[mean_z, log_var_z])

    def __call__(self, x):
        return self.model(x)

    def get_inputs(self):
        return self.inputs

    def save(self):
        self.model.save(filepath=self.cfg.encoder_z_path)

    def load(self):
        if self.model is None:
            self.model = load_model(self.cfg.encoder_z_path)

    def predict(self, inputs):
        return self.model.predict(inputs)


# class Simple_Z:
#     def __init__(self, cfg: Config, batch_size):
#         self.cfg = cfg
#         self.inputs = None
#         self.model = None
#         self.batch_size = batch_size
#
#     def init_net(self):
#         cfg = self.cfg
#         z_mean = None
#         z_log_var = None
#         cell = keras.layers.LSTMCell(cfg.z_size, name="cell")
#         mean_dense = keras.layers.Dense(cfg.z_size)
#         log_var_dense = keras.layers.Dense(cfg.z_size)
#         state = tf.zeros(shape=[self.batch_size, cfg.z_size])  # [-1, 32]
#         z_t = tf.zeros([self.batch_size, cfg.z_size])  # [-1, 32]
#         for i in range(cfg.frame):
#             h_t, state = cell(z_t, state)  # [-1, 32]
#             mean = mean_dense(h_t)  # [-1, 32]
#             log_var = log_var_dense(h_t)  # [-1, 32]
#             z_t = reparameterize(mean, log_var, cfg.training)  # [-1, 32]
#             if z_mean is None:
#                 z_mean = tf.reshape(mean, [-1, 1, cfg.z_size])
#                 z_log_var = tf.reshape(log_var, [-1, 1, cfg.z_size])
#             else:
#                 z_mean = tf.concat((z_mean, tf.reshape(mean, [-1, 1, cfg.z_size])), axis=1)
#                 z_log_var = tf.concat((z_log_var, tf.reshape(log_var, [-1, 1, cfg.z_size])), axis=1)
#         self.model = keras.Model(inputs=None, outputs=[z_mean, z_log_var])
#
#     def __call__(self):
#         return self.model(None)
#
#     def get_inputs(self):
#         return self.inputs
#
#     def save(self):
#         self.model.save(filepath=self.cfg.simple_z_path)
#
#     def load(self):
#         if self.model is None:
#             self.model = load_model(self.cfg.simple_z_path)
#
#     def predict(self, inputs):
#         return self.model.predict(inputs)


class Decoder_Frame:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.inputs = None
        self.model = None

    def init_net(self):
        cfg = self.cfg
        activation = tf.nn.elu
        f = keras.Input(shape=[cfg.f_size], dtype=tf.float32)  # [-1, 256]
        z = keras.Input(shape=[cfg.frame, cfg.z_size], dtype=tf.float32)  # [-1, 8, 32]
        new_f = keras.layers.Reshape([1, cfg.f_size])(f)  # [-1, 1, 256]
        new_f = tf.tile(new_f, [1, config.frame, 1])  # [-1, 8, 256]
        zf = tf.concat((new_f, z), axis=2)  # [-1, 8, 256 + 32]
        x = tf.reshape(zf, [-1, cfg.f_size + cfg.z_size])  # [-1, 256 + 32]
        x = keras.layers.Dense(8 * 4096)(x)  # [-1, 8 * 1024]
        x = keras.layers.Reshape([8, 4096])(x)  # [-1, 8, 1024]
        base_filters = 4096
        for i in range(4):
            # [-1, 8, 4096] -> [-1, 16, 2048] -> [-1, 32, 1024] -> [-1, 64, 512] -> [-1, 128, 256]
            base_filters /= 2
            x = keras.layers.Conv1DTranspose(filters=base_filters,
                                             kernel_size=3,
                                             strides=2,
                                             activation=activation,
                                             padding="same")(x)
        y = keras.layers.Dense(cfg.channel)(x)  # [-1, 128, 40]
        y = tf.reshape(y, [-1, cfg.frame, cfg.input_size, cfg.channel])  # [-1, 8, 128, 40]
        self.model = keras.Model(inputs=[f, z], outputs=y)

    def __call__(self, x):
        return self.model(x)

    def get_inputs(self):
        return self.inputs

    def save(self):
        self.model.save(filepath=self.cfg.decoder_frame_path)

    def load(self):
        if self.model is None:
            self.model = load_model(self.cfg.decoder_frame_path)

    def predict(self, inputs):
        return self.model.predict(inputs)


class VAE:
    def __init__(self, c: Config):
        self.cfg = c
        self.encoder_frame = Encoder_Frame(c)
        self.decoder_frame = Decoder_Frame(c)
        self.encoder_f = Encoder_F(c)
        self.encoder_z = Encoder_Z(c)
        # self.simple_z = Simple_Z(c, c.batch_size)
        self.vae = None

    def init_net(self):
        self.encoder_frame.init_net()
        self.decoder_frame.init_net()
        self.encoder_f.init_net()
        self.encoder_z.init_net()
        # self.simple_z.init_net()

        inputs = self.encoder_frame.get_inputs()
        vec = self.encoder_frame(inputs)
        mean_f, log_var_f = self.encoder_f(vec)
        f = reparameterize_tf(mean_f, log_var_f, self.cfg.training)
        mean_z, log_var_z = self.encoder_z([vec, f])
        z = reparameterize_tf(mean_z, log_var_z, self.cfg.training)
        outputs = self.decoder_frame([f, z])
        # self.simple_z()
        self.vae = keras.Model(inputs=inputs, outputs=outputs)
        self.vae.summary()
        self.vae.add_loss(self.loss(inputs, outputs, mean_f, log_var_f, mean_z, log_var_z))

    def loss(self, inputs, outputs, mean_f, log_var_f, mean_z, log_var_z):
        # [-1, 8, 128, 40]
        mes = tf.reduce_mean(tf.reduce_sum(tf.square(inputs - outputs), axis=(1, 2, 3)))
        # f : [-1, 256]
        kld_f = -0.5 * tf.reduce_mean(1 + log_var_f - tf.pow(mean_f, 2) - tf.exp(log_var_f), 1)
        # z
        # simple_z_mean, simple_z_log_var = self.simple_z()
        # z_post_var = tf.exp(log_var_z)
        # z_prior_var = tf.exp(simple_z_log_var)
        # kld_z = tf.reduce_mean(0.5 * tf.reduce_sum(simple_z_log_var - log_var_z + (z_post_var + tf.pow(mean_z - simple_z_mean, 2)) / z_prior_var - 1, axis=(1, 2)))
        loss = mes + kld_f
        return loss

    def train(self):
        cfg = self.cfg
        self.init_net()
        self.vae.compile(optimizer=keras.optimizers.Adam(learning_rate=cfg.lr))
        callbacks = tf.keras.callbacks.TensorBoard(log_dir=cfg.log_dir)
        data, marker = cfg.get_ds_train()  # data: [-1, 8, 128, 22]
        history = self.vae.fit(data,
                               batch_size=cfg.batch_size,
                               epochs=cfg.epoch,
                               callbacks=callbacks,
                               validation_split=0.1)
        print("history:", history.history)
        self.save()
        outputs = self.vae.predict(data)
        print("train_pccd:", math_pccd(data, outputs))
        # test_outputs = self.vae.predict(cfg.get_ds_test())
        # print("train_pccd:", math_pccd(cfg.get_ds_test(), test_outputs))

    def predict(self):
        cfg = self.cfg
        data, marker = cfg.get_ds_train()  # data: [-1, 8, 128, 22]
        self.vae = load_model(cfg.model_path)
        outputs = self.vae.predict(data)
        print("train_pccd:", math_pccd(data, outputs))

    def save(self):
        self.vae.save(filepath=self.cfg.model_path)
        self.encoder_frame.save()
        self.encoder_f.save()
        self.encoder_z.save()
        # self.simple_z.save()
        self.decoder_frame.save()

    def re_view(self):
        cfg = self.cfg
        cfg.training = False
        # simple = cfg.get_ds_test()  # 28
        simple, marker = cfg.get_ds_train()  # data: [-1, 8, 128, 22]
        marker = np.array(marker)
        x = []
        for i in range(6):
            index = np.where(marker == i)[0][0]
            x.append(simple[index])
        x = np.array(x)
        # x = simple[8: 10]  # [10, 8, 128, 40]
        # marker = marker[8: 10]
        print("x: ", np.shape(x))
        self.vae = load_model(cfg.model_path)
        y = self.vae.predict(x)

        size = cfg.frame * cfg.input_size  # 8 * 128
        i = 1
        for a, b in zip(x, y):
            print("a: ", np.shape(a))
            print("b: ", np.shape(b))
            pccd = math_pccd(a, b)
            print(pccd)
            a = np.reshape(a, [size, cfg.channel])  # 1024, 22
            b = np.reshape(b, [size, cfg.channel])  # 1024, 22
            plt.subplot(12, 1, i)
            plt.plot([j for j in range(size)], a)

            plt.subplot(12, 1, i + 1)
            plt.plot([j for j in range(size)], b)
            i += 2
        plt.savefig("image/{name}/review_f5.png".format(name=cfg.get_name()))

    def features_change(self):
        cfg = self.cfg
        cfg.training = False
        simple, marker = cfg.get_ds_test()  # 28
        x1 = simple[8]  # [8, 128, 40]
        x2 = simple[18]  # [8, 128, 40]
        self.encoder_frame.load()
        self.encoder_f.load()
        self.encoder_z.load()
        self.decoder_frame.load()

        vec_1 = self.encoder_frame.predict(np.array([x1]))  # [1, 8, 1024]
        vec_2 = self.encoder_frame.predict(np.array([x2]))  # [1, 8, 1024]

        mean_f_1, log_var_f_1 = self.encoder_f.predict(vec_1)
        mean_f_2, log_var_f_2 = self.encoder_f.predict(vec_2)

        mean_z_1, log_var_z_1 = self.encoder_z.predict([vec_1, mean_f_1])
        mean_z_2, log_var_z_2 = self.encoder_z.predict([vec_2, mean_f_2])

        f1_z2 = self.decoder_frame.predict([mean_f_1, mean_z_2])  # [1, 8, 128, 40]
        f2_z1 = self.decoder_frame.predict([mean_f_2, mean_z_1])

        size = cfg.frame * cfg.input_size
        x1 = np.reshape(x1, [size, cfg.channel])
        x2 = np.reshape(x2, [size, cfg.channel])
        f1_z2 = np.reshape(f1_z2, [size, cfg.channel])  # [8, 128, 40]
        f2_z1 = np.reshape(f2_z1, [size, cfg.channel])

        plt.subplot(4, 1, 1)
        plt.plot([j for j in range(size)], x1)

        plt.subplot(4, 1, 2)
        plt.plot([j for j in range(size)], x2)

        plt.subplot(4, 1, 3)
        plt.plot([j for j in range(size)], f1_z2)

        plt.subplot(4, 1, 4)
        plt.plot([j for j in range(size)], f2_z1)
        plt.savefig("image/{name}/features_change.png".format(name=cfg.get_name()))

    def f_zero_z(self):
        cfg = self.cfg
        cfg.training = False
        simple, marker = cfg.get_ds_test()  # 28
        # todo
        marker = np.array(marker)
        x = []
        for i in range(6):
            index = np.where(marker == i)[0][0]
            x.append(simple[index])
        x = np.array(x)

        # x1 = simple[8]  # [8, 128, 40]
        # x2 = simple[18]  # [8, 128, 40]
        self.encoder_frame.load()
        self.encoder_f.load()
        self.encoder_z.load()
        self.decoder_frame.load()

        vec_1 = self.encoder_frame.predict(x)  # [1, 8, 1024]
        # vec_2 = self.encoder_frame.predict([x2])  # [1, 8, 1024]

        mean_f_1, log_var_f_1 = self.encoder_f.predict(vec_1)  # [1, 256]
        # mean_f_2, log_var_f_2 = self.encoder_f.predict(vec_2)

        mean_z_1, log_var_z_1 = self.encoder_z.predict([vec_1, mean_f_1])  # [1, 8, 32]
        # mean_z_2, log_var_z_2 = self.encoder_z.predict([vec_2, mean_f_2])

        f = np.zeros(shape=[6, cfg.f_size])
        # f1_z2 = self.decoder_frame.predict([f, mean_z_2])  # [1, 8, 128, 40]
        y = self.decoder_frame.predict([f, mean_z_1])

        # x1 = np.reshape(x1, [cfg.frame * cfg.input_size, cfg.channel])  # [8 * 128, 40]
        # x2 = np.reshape(x2, [cfg.frame * cfg.input_size, cfg.channel])  # [8 * 128, 40]
        # f1_z2 = np.reshape(f1_z2, [cfg.frame * cfg.input_size, cfg.channel])
        # f2_z1 = np.reshape(f2_z1, [cfg.frame * cfg.input_size, cfg.channel])
        i = 1
        for x_i, y_i in zip(x, y):  # [8, 128, 40]
            x_i = np.reshape(x_i, [cfg.frame * cfg.input_size, cfg.channel])
            plt.subplot(12, 1, i)
            plt.plot([j for j in range(cfg.frame * cfg.input_size)], x_i)
            i += 1
            y_i = np.reshape(y_i, [cfg.frame * cfg.input_size, cfg.channel])
            plt.subplot(12, 1, i)
            plt.plot([j for j in range(cfg.frame * cfg.input_size)], y_i)
            i += 1
        plt.savefig("image/{name}/f0.png".format(name=cfg.get_name()))
        # plt.subplot(4, 1, 1)
        # plt.plot([j for j in range(cfg.frame * cfg.input_size)], x1)
        #
        # plt.subplot(4, 1, 2)
        # plt.plot([j for j in range(cfg.frame * cfg.input_size)], x2)
        #
        # plt.subplot(4, 1, 3)
        # plt.plot([j for j in range(cfg.frame * cfg.input_size)], f1_z2)
        #
        # plt.subplot(4, 1, 4)
        # plt.plot([j for j in range(cfg.frame * cfg.input_size)], f2_z1)
        # plt.savefig("image/{name}/f0_z.png".format(name=cfg.get_name()))


if __name__ == '__main__':
    config = Config()
    # simple, marker = config.get_ds_train()  # data: [-1, 8, 128, 22]
    # marker = np.array(marker)
    # x = []
    # for i in range(6):
    #     index = np.where(marker == i)[0][0]
    #     x.append(simple[index])
    # x = np.array(x)

    vae = VAE(config)
    # vae.predict()
    # vae.train()
    # vae.re_view()
    vae.features_change()
    vae.f_zero_z()