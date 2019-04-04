from abc import ABC, abstractmethod
from typing import Dict

from torch import nn


class BaseModel(ABC):
    def __init__(self):
        self._model_states: Dict[str, nn.Module] = {}

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    def load_state(self, model_state):
        for state_key in model_state:
            state_dict = model_state[state_key]
            self._model_states[state_key].load_state_dict(state_dict)
