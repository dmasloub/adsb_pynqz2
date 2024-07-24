from sklearn.discriminant_analysis import StandardScaler
from sklearn.pipeline import Pipeline
from tensorflow.keras.layers import (
    Input, 
)
from tensorflow.keras.models import (
    Model, 
)
from qkeras import (
    QDense, 
    QActivation, 
    quantized_bits, 
    quantized_relu
)

from utils.constants import (
    PERCENT, 
    BEGIN_STEP, 
    FREQUENCY
)

from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule

def get_quantized_autoencoder_model(input_dim, encoding_dim, bits=6, integer=0, alpha=1):
    """
    Create a quantized autoencoder model.
    
    Parameters:
    input_dim (int): Dimension of the input data.
    encoding_dim (int): Dimension of the encoded representation.
    bits (int): Number of bits for quantization. Default is 6.
    integer (int): Number of integer bits for quantization. Default is 0.
    alpha (float): Scaling factor for quantization. Default is 1.
    
    Returns:
    Model: A Keras Model representing the quantized autoencoder.
    """
    try:
        input_layer = Input(shape=(input_dim,))
        
        hidden_layer = QDense(
            encoding_dim,
            kernel_quantizer=quantized_bits(bits, integer, alpha=alpha),
            bias_quantizer=quantized_bits(bits, integer, alpha=alpha),
            activation="relu"
        )(input_layer)
        
        output_layer = QDense(
            input_dim,
            kernel_quantizer=quantized_bits(bits, integer, alpha=alpha),
            bias_quantizer=quantized_bits(bits, integer, alpha=alpha),
            activation="relu"
        )(hidden_layer)
        
        autoencoder = Model(inputs=input_layer, outputs=output_layer)
        
        return autoencoder
    except Exception as e:
        print(f"Error creating quantized autoencoder model: {e}")
        raise
    
pruning_params = {
    "pruning_schedule": pruning_schedule.ConstantSparsity(PERCENT, begin_step=BEGIN_STEP, frequency=FREQUENCY)
}

autoencoder_model_feature_pipeline = Pipeline(
    steps=[('normalize', StandardScaler())]
)
