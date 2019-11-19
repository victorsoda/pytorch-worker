import sys
sys.path.append('/home/cyd/pytorch-worker')
import json
from torch.utils.data import Dataset
import os
import numpy as np
import argparse
from config_parser import create_config


def unpickle(filename):
    import pickle
    with open(filename, 'rb') as fo:
        dic = pickle.load(fo, encoding='bytes')
    return dic


class Cifar10Dataset(Dataset):
    def __init__(self, config, mode, *args, **params):
        self.config = config
        self.mode = mode
        self.data_path = config.get("data", "%s_data_path" % mode)
        
        self.file_list = config.get("data", "%s_file_list" % mode).split(' ')
        self.data_list = []
        self.load_mem = config.getboolean("data", "load_into_mem")

        if self.load_mem:
            for filename in self.file_list:
                dic = unpickle(os.path.join(self.data_path, filename))
                data_num = dic[b'data'].shape[0]
                for i in range(data_num):
                    sample = {"data": dic[b'data'][i], "label":dic[b'labels'][i]}
                    self.data_list.append(sample)
            # print("len:", len(self.data_list))

    def __getitem__(self, item):
        if self.load_mem:
            return {
                "data": self.data_list[item]["data"],
                "label": self.data_list[item]["label"]
            }
        else:
            return {
                "data": cv2.imread(os.path.join(self.prefix, self.data_list[item]["path"])),
                "label": self.data_list[item]["label"]
            }

    def __len__(self):
        return len(self.data_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help="specific config file", required=True)
    parser.add_argument('--gpu', '-g', help="gpu id list")
    parser.add_argument('--checkpoint', help="checkpoint file path")
    args = parser.parse_args()

    configFilePath = args.config
    config = create_config(configFilePath)
    mode = "test"

    x = Cifar10Dataset(config, mode)
