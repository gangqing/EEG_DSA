import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
# from model_2 import Config
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""
基于Data4的基础上，做以下改动：
1、舍弃99标签以前的数据(包括99标签) 、 91号以后的标签(包括91号)
2、全局Z标准化；
3、去掉异常数据，z值大于4.0或者小于-4.0的值当作是异常值；
"""


def show(data, marker, path):
    plt.subplot(2, 1, 1)
    plt.plot([i for i in range(len(data))], data)

    plt.subplot(2, 1, 2)
    plt.plot([i for i in range(len(marker))], marker)

    plt.savefig(path)


def deal_data(data, marker):
    # 舍弃99标签以前的数据(包括99标签) 、 91号以后的标签(包括91号)
    index_99 = np.max(np.where(marker == 99)[0])
    index_91 = np.min(np.where(marker == 91)[0])
    marker = marker[index_99 + 1: index_91]
    data = data[index_99 + 1: index_91]
    # 舍弃91号数据
    # index_91 = np.where(marker == 91)[0]
    # data = np.delete(data, index_91, axis=0)
    # marker = np.delete(marker, index_91, axis=0)
    return data, marker


class Data:
    def __init__(self, cfg):
        self.cfg = cfg
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
        cfg = self.cfg
        size = cfg.frame * cfg.input_size  # 8 * 128
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
            num = len(data) // size
            if num == 0:
                continue
            data = data[: num * size]
            data = np.reshape(data, [-1, cfg.frame, cfg.input_size, cfg.channel])
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
    # cfg = Config()
    ds = Data(None)
