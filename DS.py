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
        self.simple = []  # [1280, 40, 8064]
        self.load_file()
        self.op = np.random.randint(0, self.num_examples)

    def load_file(self):
        file_name_list = os.listdir(self.file_path)
        file_name_list.sort()
        size = self.config.input_size
        simple = []
        for file_name in file_name_list:
            file = cPickle.load(open(self.file_path + file_name, 'rb'), encoding="bytes")
            data = file["data".encode(encoding="utf-8")]  # [40, 40, 8064]
            # 归一化
            data = self.pre_processing_normal(data)
            for i in range(len(data)):
                ds = data[i, :self.config.frame, :]  # [40, 8064]
                num = np.shape(ds)[-1] // size
                for j in range(num):
                    sub_data = ds[:, size * j: size * j + size]  # [40, size]
                    simple.append(sub_data)  # [-1, 40, size]
            # 只加载一个文件
            break
        self.simple = np.array(simple)
        self.simple[np.isnan(self.simple)] = 0.0
        self.simple = self.simple.tolist()

    def pre_processing_normal(self, ds):
        # mean = np.reshape(np.mean(ds, axis=1), [len(ds), 1])  # [40, 1]
        # std = np.reshape(np.std(ds, axis=1), [len(ds), 1])  # [40, 1]
        # ds = (ds - mean) / std
        self.mean = np.mean(ds)
        self.std = np.std(ds)
        return (ds - self.mean) / self.std

    def to_data(self, ds):
        pass

    @property
    def num_examples(self):
        return len(self.simple)

    def next_batch(self, batch_size):
        next_op = self.op + batch_size
        if next_op > self.num_examples:
            result = self.simple[self.op:]
            next_op -= self.num_examples
            result.extend(self.simple[:next_op])
        else:
            result = self.simple[self.op: next_op]
            self.op = next_op
        return [result]
