import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.accuracy_init import init_accuracy_function


class NaiveCNN(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(NaiveCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.criterion = nn.CrossEntropyLoss()
        self.accuracy_function = init_accuracy_function(config, *args, **params)

    def init_multi_gpu(self, device, config, *args, **params):
        # self.bert = nn.DataParallel(self.bert, device_ids=device)
        pass

    def forward(self, data, config, gpu_list, acc_result, mode):
        x = data["input"]

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = self.fc3(x)

        if "label" in data.keys():
            label = data["label"]
            loss = self.criterion(y, label)
            acc_result = self.accuracy_function(y, label, config, acc_result)
            return {"loss": loss, "acc_result": acc_result}

        return {}
