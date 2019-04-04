import importlib
import json

import falcon
import numpy as np
import toml
import torch

from model.base_model import BaseModel


class ModelResource:
    def __init__(self, model: BaseModel):
        self._model = model

    def on_post(self, req, resp):
        try:
            req_body = req.stream.read()
            input_data = torch.from_numpy(np.array(json.loads(req_body)['input']))
            output = self._model.predict(input_data)
        except RuntimeError as e:
            resp.status = falcon.HTTP_500
            resp.body = 'Input error:\n' + str(e)
        else:
            resp.status = falcon.HTTP_200
            resp.content_type = 'text/html'
            resp.body = str(output)


def build_app(*args, **kwargs):
    model_toml = kwargs['model']
    model_conf_file = model_toml
    model_conf = toml.load(model_conf_file)
    module_name = model_conf['model_module']
    class_name = model_conf['model_class']
    model_state_file = model_conf['model_state_file']

    class_ = getattr(importlib.import_module(module_name), class_name)
    model_obj: BaseModel = class_()

    model_state = torch.load(model_state_file)
    model_obj.load_state(model_state)

    app = falcon.API()
    app.req_options.auto_parse_form_urlencoded = True
    app.add_route('/', ModelResource(model_obj))

    return app
