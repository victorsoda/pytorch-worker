import json
from torch.utils.data import Dataset
import os
import numpy as np
import argparse
from config_parser import create_config
from tools.standard_scaler import StandardScaler


class METRDataset(Dataset):
    def __init__(self, config, mode, *args, **params):
        self.config = config
        self.mode = mode
        self.data_path = config.get("data", "%s_data_path" % mode)
        
        self.file_list = config.get("data", "%s_file_list" % mode).split(' ')
        self.data_list = {}
        self.load_mem = config.getboolean("data", "load_into_mem")
        self.scaler_path = config.get("data", "scaler_path")

        if self.load_mem:
            for filename in self.file_list:
                cat_data = np.load(os.path.join(self.data_path, filename))
                print("x.shape", cat_data['x'].shape, "y.shape", cat_data['y'].shape)
                self.data_list['x'] = cat_data['x']
                self.data_list['y'] = cat_data['y']
            if self.mode == "train":
                mean_std = np.array([self.data_list['x'][..., 0].mean(), self.data_list['x'][..., 0].std()])
                print("mean and std of training set:", mean_std)
                np.save(self.scaler_path, mean_std)


    def __getitem__(self, item):
        if self.load_mem:
            return {
                "x": self.data_list["x"][item],
                "y": self.data_list["y"][item]
            }
        else:
            return {
                # "data": cv2.imread(os.path.join(self.prefix, self.data_list[item]["path"])),
                # "label": self.data_list[item]["label"]
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
    mode = "train"

    x = METRDataset(config, mode)