import os

from ...utils import import_modules


SENSOR_ENCODER_FACTORY = {}


def SensorEncoderFactory(encoder_name):
    if encoder_name is None or encoder_name == "":
        return lambda *args, **kwargs: None
    model = None
    for name, cls in SENSOR_ENCODER_FACTORY.items():
        if name.lower() in encoder_name.lower():
            model = cls
            break
    assert model, f"{encoder_name} is not registered"
    return model


def register_sensor_encoder(name):
    def register_encoder_cls(cls):
        if name in SENSOR_ENCODER_FACTORY:
            return SENSOR_ENCODER_FACTORY[name]
        SENSOR_ENCODER_FACTORY[name] = cls
        return cls
    return register_encoder_cls


models_dir = os.path.dirname(__file__)
import_modules(models_dir, "tinyllava.model.sensor_encoder")
