import importlib

import falcon
import toml
import torch

from model.base_model import BaseModel


class ModelResource:
    def __init__(self, model: BaseModel, model_state_file, use_device='cpu'):
        if use_device == 'gpu':
            model_state = torch.load(model_state_file, map_location=torch.device('cpu'))
        else:
            model_state = torch.load(model_state_file)
        model.load_state(model_state)


def build_app(*args, **kwargs):
    model_name = kwargs['model']
    model_conf_file = f'{model_name}.toml'
    model_conf = toml.load(model_conf_file)
    module_name = model_conf['model_module']
    class_name = model_conf['model_class']
    model_state_file = model_conf['model_state_file']

    class_ = getattr(importlib.import_module(module_name), class_name)
    model_obj = class_()

    app = falcon.API()
    app.req_options.auto_parse_form_urlencoded = True
    app.add_route('/', ModelResource(model_obj, model_state_file))

    return app
