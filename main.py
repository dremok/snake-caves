import torch


class ModelResource:
    def __init__(self, model_file, state_dicts, use_device='cpu'):
        if use_device == 'gpu':
            checkpoint = torch.load(model_file, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(model_file)

# TODO: Load model state from configured state dicts
