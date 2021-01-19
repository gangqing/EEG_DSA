import tensorflow as tf
from tensorflow import keras
import numpy as np
import scipy.io as sio
from model_2 import VAE


class DNN:
    def __init__(self):
        inputs = keras.Input([22, 128], dtype=tf.float32)  # [-1, 22, 128]
        d1 = keras.layers.BatchNormalization()(inputs)
        d1 = keras.layers.Dense(32,
                                activation=tf.nn.relu,
                                activity_regularizer=keras.regularizers.L1(),
                                kernel_regularizer=keras.regularizers.L2())(d1)  # [-1, 22, 32]

        d2 = keras.layers.BatchNormalization()(d1)
        d2 = keras.layers.Dense(8, activation=tf.nn.relu)(d2)  # [-1, 22, 8]

        f = keras.layers.Flatten()(d2)  # [-1, 176]

        d3 = keras.layers.BatchNormalization()(f)
        d3 = keras.layers.Dense(32, activation=tf.nn.relu)(d3)  # [-1, 32]

        d4 = keras.layers.BatchNormalization()(d3)
        outputs = keras.layers.Dense(6, activation=tf.nn.softmax)(d4)  # [-1, 6]

        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                           loss=tf.keras.losses.sparse_categorical_crossentropy,  # 交叉熵
                           metrics=['accuracy'])  # 准确度
        self.model.summary()


def deal_data(data, marker):
    # 舍弃99标签以前的数据(包括99标签) 、 91号以后的标签(包括91号)
    index_99 = np.max(np.where(marker == 99)[0])
    index_91 = np.min(np.where(marker == 91)[0])
    marker = marker[index_99 + 1: index_91]
    data = data[index_99 + 1: index_91]
    return data, marker


class Data:
    def __init__(self):
        self.simple_path = "data_f5/5F-SubjectB-160311-5St-SGLHand-HFREQ.mat"
        self.source_data = None  # [-1, 22]
        self.source_marker = None  # [-1, 1]
        self.mean = None
        self.std = None
        self.create_data()
        self.simple()

    def create_data(self):
        data, marker = self.load_mat_data()
        data, self.source_marker = deal_data(data, marker)  # marker: [-1, 1]
        # 归一化
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        self.source_data = (data - self.mean) / self.std  # [-1, 22]

    def simple(self):
        """
        simple : [-1, 8, 128, 22]
        """
        markers = np.reshape(self.source_marker, [-1])
        start_index = 0
        current_marker = None
        simple = None
        label = []
        for index, marker in enumerate(markers):  # marker: 0 or 1 or 2 or 3 or 4 or 5
            if current_marker is None:
                current_marker = marker
                continue
            if current_marker == marker:
                continue
            data = self.source_data[start_index: index]  # [-1, 22]
            num = len(data) // 128
            if num == 0:
                continue
            data = data[: num * 128]  # [-1, 22]
            data = np.reshape(data, [-1, 128, 22])  # [-1, 128, 22]
            data = np.transpose(data, [0, 2, 1])  # [-1, 22, 128]
            if simple is None:
                simple = data
            else:
                simple = np.concatenate((simple, data), axis=0)
            for i in range(len(data)):
                label.append(current_marker)
            # 更新参数
            start_index = index
            current_marker = marker
        return simple, label

    def load_mat_data(self):
        """
        加载数据
        :return:
        data: [-1, 22]
        marker:
        0 : 无操作;
        1 - 5 ：对应手指操作;
        91 ：between;
        92 ：end;
        99 ：start
        """
        file = sio.loadmat(self.simple_path)
        data = file["o"]["data"][0][0]  # [-1, 22]
        data = np.array(data).astype(np.float)

        marker = file["o"]["marker"][0][0]  # [-1, 1]
        marker = np.array(marker).astype(np.int)
        return data, marker

    def to_value(self, data):
        if self.mean is None:
            self.get_param()
        return data * self.std + self.mean

    def get_param(self):
        data, source_marker = self.load_mat_data()
        # 归一化
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)


if __name__ == '__main__':
    dnn = DNN()
    DS = Data()
    simple, marker = DS.simple()
    marker = np.array(marker)
    # 舍弃0标签的数据
    index_0 = np.where(marker == 0)[0]
    simple = np.delete(simple, index_0, axis=0)
    marker = np.delete(marker, index_0, axis=0)
    for i in range(1, 6):
        index_list = np.where(marker == i)[0]
        print("{i}: ".format(i=i), float(len(index_list))/len(marker))
    marker = np.reshape(marker, [-1, 1])
    history = dnn.model.fit(simple,
                            marker,
                            batch_size=100,
                            epochs=100,
                            validation_split=0.2)
