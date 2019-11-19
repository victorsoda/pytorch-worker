import json
import torch
import os
import torchvision.transforms as transforms
import argparse
from dataset.others.METR import METRDataset
from formatter.Basic import BasicFormatter
from config_parser import create_config


class METRFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)

        self.config = config
        self.mode = mode
        self.scaler_path = config.get("data", "scaler_path")

    def process(self, data, config, mode, *args, **params):
        assert os.path.isfile(self.scaler_path)
        mean, std = np.load(self.scaler_path)
        scaler =  StandardScaler(mean=mean, std=std)
        
        _x = []
        _y = []
        for sample in data:
            x = torch.tensor(sample['x'][..., 0]).float()
            y = torch.tensor(sample['y'][..., 0]).float()
            _x.append(x)
            _y.append(y)
        ret = {}
        ret['x'] = torch.stack(scaler.transform(_x))
        ret['y'] = torch.stack(scaler.transform(_y))
        return ret
            
        # _input = []
        # _label = []
        # for sample in data:
        #     x = torch.tensor(sample["data"].reshape(3, 32, 32)).float()
        #     y = torch.tensor(sample["label"]).long()
        #     _input.append(x)
        #     _label.append(y)
        # return {"input": torch.stack(_input), "label": torch.stack(_label)}
        



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help="specific config file", required=True)
    parser.add_argument('--gpu', '-g', help="gpu id list")
    parser.add_argument('--checkpoint', help="checkpoint file path")
    args = parser.parse_args()

    configFilePath = args.config
    config = create_config(configFilePath)
    mode = "train"

    dataset = Cifar10Dataset(config, mode)
    data = []
    for i in range(16):
        data.append(dataset.__getitem__(i))

    form = Cifar10Formatter(config, mode)
    form.process(data, config, mode)

