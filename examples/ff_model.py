import torch
from torch import nn

from model.base_model import BaseModel


class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class FeedForwardClassificationModel(BaseModel):
    def __init__(self):
        self._simple_net = SimpleNet(784, 512, 10)
        modules = {'mnist': self._simple_net}
        super().__init__(modules)

    def train(self):
        pass

    def predict(self, input_data):
        with torch.no_grad():
            outputs = self._simple_net(input_data.float())
            _, predicted = torch.max(outputs.data, 0)
            return predicted.item()
