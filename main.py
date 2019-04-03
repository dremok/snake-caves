import torch

from model.base_model import BaseModel


class ModelResource:
    def __init__(self, model: BaseModel, model_state_file, use_device='cpu'):
        if use_device == 'gpu':
            model_state = torch.load(model_state_file, map_location=torch.device('cpu'))
        else:
            model_state = torch.load(model_state_file)
        model.load_state(model_state)
