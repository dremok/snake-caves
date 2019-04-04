# SnakeCaves
A light-weight prediction service for PyTorch models.

## How to use
1. Create a subclass of the `BaseModel` class. Implement your model as a combination of PyTorch modules, and implement the `train` and `predict` methods of your subclass. Train your model and call `save_state` to save it.
2. Create a [TOML](https://github.com/toml-lang/toml) file containing these 3 fields: model_module, model_class and model_state_file. Example:
```
model_module = "examples.ff_model"
model_class = "FeedForwardClassificationModel"
model_state_file = "examples/ff.tar"
```

`model_module` is the module containing the `model_class`, which is the name of your model class inheriting from `BaseModel`. The `model_state_file` is the file where you saved the final state of your trained model.

3. Run `gunicorn app:build_app(model='path_to_toml_file')` to start a web server wrapping the trained model in a simple API. To test it, make a POST request to `http://localhost:8000` with a JSON body containing the model input in a property named `input`.
4. ???
5. PROFIT!
