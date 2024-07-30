import hls4ml
from keras.models import load_model
from keras.utils import custom_object_scope
from qkeras import QDense, quantized_bits  # Import required components from QKeras
from src.utils.visualization import print_dict

if __name__ == "__main__":
    # Define custom objects
    custom_objects = {'QDense': QDense, 'quantized_bits': quantized_bits}

    # Load the model with custom object scope
    with custom_object_scope(custom_objects):
        config = hls4ml.converters.convert_from_config("hls_model/hls4ml_prj/hls4ml_config.yml")
        keras_model = load_model("hls_model/hls4ml_prj/keras_model.h5")
    
    hls_model = hls4ml.converters.convert_from_config(config=config)

    print_dict(config)
