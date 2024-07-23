import torch
from autogluon.tabular import TabularPredictor

# Load your model using AutoGluon
model_path = './datasets/CICIDS2017/balanced_binary/automl_search/models/NeuralNetTorch/a7a36c5f'
predictor_path = './datasets/CICIDS2017/balanced_binary/automl_search/'
predictor = TabularPredictor.load(predictor_path)

# Try to access the internal PyTorch model
model = None

# Check various possible attributes to find the PyTorch model
if hasattr(predictor._trainer, 'model_best'):
    model = predictor._trainer.model_best
elif hasattr(predictor._trainer, 'model'):
    model = predictor._trainer.model

# Ensure the model is callable
if model is None or not isinstance(model, torch.nn.Module):
    raise TypeError("The loaded model is not a PyTorch model")

# Inspect the model to determine the expected input shape
print(model)

# Function to print the input and output shapes
def print_input_output_shape(model, input_shape):
    dummy_input = torch.randn(input_shape)
    print(f"Input shape: {dummy_input.shape}")
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Output shape: {output.shape}")

# Calling the function with an initial guess of the input shape
initial_guess = (1, 3, 224, 224)  # Modify this according to your model
print_input_output_shape(model, initial_guess)

