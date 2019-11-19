from .cv.NaiveCNN import NaiveCNN

model_list = {
    "NaiveCNN": NaiveCNN
}


def get_model(model_name):
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError
