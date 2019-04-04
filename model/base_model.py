from abc import ABC, abstractmethod
from typing import Dict

import torch
from torch import nn


class BaseModel(ABC):
    def __init__(self, modules: Dict[str, nn.Module]):
        self._modules: Dict[str, nn.Module] = modules

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self, input_data):
        pass

    def load_state(self, model_state):
        for state_key in model_state:
            state_dict = model_state[state_key]
            self._modules[state_key].load_state_dict(state_dict)

    def save_state(self, model_state_file):
        torch.save({module_name: module.state_dict() for module_name, module in self._modules.items()},
                   model_state_file)
