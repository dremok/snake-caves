from torch import nn

from model.base_model import BaseModel


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
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
        modules = {'net': Net(784, 512, 10)}
        super().__init__(modules)

    def train(self):
        pass

    def predict(self):
        pass
