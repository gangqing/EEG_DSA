import _pickle as cPickle
import os
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class DS:
    def __init__(self, config):
        self.config = config
        self.file_path = "data/"
        self.mean = None
        self.std = None
        self.train_data = []  # [1280, 40, 8064]
        self.test_data = []
        self.load_file()

    def load_file(self):
        file_name_list = os.listdir(self.file_path)
        file_name_list.sort()
        simple = []
        inputs_size = self.config.input_size  # 128
        frame = self.config.frame  # 8
        size = inputs_size * frame  # 1024
        count = 0
        for file_name in file_name_list:
            file = cPickle.load(open(self.file_path + file_name, 'rb'), encoding="bytes")
            data = file["data".encode(encoding="utf-8")]  # [40, 40, 8064]
            data = np.transpose(data, [0, 2, 1])  # [40, 8064, 40]
            for i in range(len(data)):
                ds = data[i]  # [8064, 40]
                num = 8064 // size  # 7
                for j in range(num):
                    sub_data = ds[size * j: size * (j + 1), :]  # [128 * 8, 40]
                    sub_data = np.reshape(sub_data, [frame, inputs_size, 40])  # [8, 128, 40]
                    simple.append(sub_data)  # [-1, 8, 128, 40]
            count += 1
            if count >= 1:
                break
        # Z标准化
        self.mean = np.mean(simple)
        self.std = np.std(simple)
        simple = (simple - self.mean) / self.std  # [-1, 8, 128, 40]
        # 构造样本
        op = int(len(simple) / 10)
        self.test_data = np.array(simple[:op])
        self.train_data = np.array(simple[op:])


    def to_data(self, ds):
        pass


