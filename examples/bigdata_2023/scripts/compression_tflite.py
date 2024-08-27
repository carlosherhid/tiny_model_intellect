import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.onnx
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import onnx
import matplotlib.pyplot as plt
import onnxruntime as ort
#from onnx_tf.backend import prepare
import tensorflow as tf
from autogluon.tabular import TabularPredictor

class CSVDataset(Dataset):
    def __init__(self, file_path, label_column=None):
        self.data = pd.read_csv(file_path)
        self.label_column = label_column
        if self.label_column and self.label_column in self.data.columns:
            self.labels = self.data[self.label_column].values
            self.data = self.data.drop(columns=[self.label_column])
        else:
            self.labels = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx].values.astype(np.float32)
        if self.labels is not None:
            label = self.labels[idx]
            return sample, label
        return sample

def evaluate_model(predictor, X, y):
    y_pred = predictor.predict(X)
    accuracy = accuracy_score(y, y_pred)
    return accuracy

def evaluate_onnx_model(onnx_model_path, test_loader, device):
    ort_session = ort.InferenceSession(onnx_model_path)
    y_true = []
    y_pred = []

    for inputs, labels in test_loader:
        inputs = inputs.to(device).numpy()
        # Ensure the input sample has the correct shape
        if inputs.shape[1] == 62:
            inputs = np.pad(inputs, ((0, 0), (0, 5)), mode='constant', constant_values=0)
        ort_inputs = {ort_session.get_inputs()[0].name: inputs}
        ort_outs = ort_session.run(None, ort_inputs)
        predictions = np.argmax(ort_outs[0], axis=1)
        y_true.extend(labels.numpy())
        y_pred.extend(predictions)

    accuracy = accuracy_score(y_true, y_pred)
    return accuracy

def convert_pytorch_to_onnx(model, input_sample, onnx_path):
    torch.onnx.export(model, input_sample, onnx_path, export_params=True, opset_version=10, do_constant_folding=True, input_names=['input'], output_names=['output'])

# Custom PyTorch model with detailed debugging
class CustomModel(torch.nn.Module):
    def __init__(self, model):
        super(CustomModel, self).__init__()
        self.model = model

    def forward(self, x):
        print(f"Initial input shape: {x.shape}")
        for name, layer in self.model.named_children():
            x = layer(x)
            print(f"After layer {name}: {x.shape}")
            if isinstance(layer, torch.nn.BatchNorm1d):
                print(f"BatchNorm layer {name} expects {layer.running_mean.shape[0]} features.")
        return x

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    model_directory = args.model_directory
    model_name = os.path.basename(model_directory)
    predictor_path = os.path.abspath(os.path.join(model_directory, "../../../"))
    compressed_model_path = os.path.abspath(os.path.join(args.output_directory, "compressed_models"))
    os.makedirs(compressed_model_path, exist_ok=True)

    # Load AutoGluon model
    predictor = TabularPredictor.load(predictor_path)
    model = predictor._trainer.load_model(predictor.get_model_best())
    
    # Print the model architecture details
    print(f"Model architecture:\n{model.model}")
    for name, layer in model.model.named_children():
        print(f"Layer {name}: {layer}")
        if isinstance(layer, torch.nn.BatchNorm1d):
            print(f"BatchNorm layer {name} expects {layer.running_mean.shape[0]} features.")

    # Extract PyTorch model from AutoGluon model
    torch_model = CustomModel(model.model)
    torch_model.to(device)
    torch_model.eval()

    # Create DataLoader for test data
    test_dataset = CSVDataset(args.test_data, label_column='Label')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Get a batch of input samples
    input_sample, _ = next(iter(test_loader))
    input_sample = input_sample.to(device)
    
    # Ensure the input sample has the correct shape
    if input_sample.shape[1] == 62:
        # Add 5 zero features to match the expected input size of 67
        input_sample = torch.nn.functional.pad(input_sample, (0, 5), "constant", 0)
    
    print(f"Prepared input sample shape: {input_sample.shape}")

    onnx_path = os.path.join(compressed_model_path, "model.onnx")
    convert_pytorch_to_onnx(torch_model, input_sample, onnx_path)

    # Evaluate AutoGluon model
    test_data = pd.read_csv(args.test_data)
    X_test = test_data.drop(columns=['Label'])
    y_test = test_data['Label']
    autogluon_accuracy = evaluate_model(predictor, X_test, y_test)

    # Evaluate ONNX model
    onnx_accuracy = evaluate_onnx_model(onnx_path, test_loader, device)

    # Print accuracies
    print(f"AutoGluon Model Accuracy: {autogluon_accuracy}")
    print(f"ONNX Model Accuracy: {onnx_accuracy}")

    # Plot accuracies
    models = ['AutoGluon', 'ONNX']
    accuracies = [autogluon_accuracy, onnx_accuracy]
    
    plt.figure(figsize=(10, 5))
    plt.bar(models, accuracies, color=['blue', 'orange'])
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.show()
"""
    # Convert ONNX model to TensorFlow
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)

    # Save the TensorFlow model
    tf_model_path = os.path.join(compressed_model_path, "model")
    tf_rep.export_graph(tf_model_path)

    # Evaluate the TensorFlow model
    evaluate_tf_model(tf_model_path, args.test_data)

def evaluate_tf_model(tf_model_path, test_data_path):
    # Load the TensorFlow model
    model = tf.saved_model.load(tf_model_path)
    infer = model.signatures["serving_default"]

    # Load test data
    test_dataset = CSVDataset(test_data_path, label_column='Label')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    predictions = []
    y_true = []

    for batch in test_loader:
        if isinstance(batch, tuple):
            input_tensor, labels = batch
            y_true.extend(labels.numpy())
        else:
            input_tensor = batch
        input_tensor = tf.convert_to_tensor(input_tensor.numpy(), dtype=tf.float32)
        batch_predictions = infer(input_tensor)['output'].numpy()
        predictions.extend(batch_predictions)

    if y_true:
        accuracy = accuracy_score(y_true, np.argmax(predictions, axis=1))
        print(f"Accuracy: {accuracy}")
    else:
        print("No ground truth labels found in the test data.")
    _summary_
    """
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compress and evaluate AutoGluon models")
    parser.add_argument('--model-directory', type=str, required=True, help="Directory of the best model (e.g., ./datasets/CICIDS2017/balanced_binary/automl_search/models/NeuralNetTorch/a7a36c5f)")
    parser.add_argument('--output-directory', type=str, required=True, help="Directory to save compressed models and results")
    parser.add_argument('--test-data', type=str, required=True, help="Path to the test data CSV file (e.g., ./datasets/CICIDS2017/balanced_binary/test.csv)")
    parser.add_argument('--architecture', type=str, choices=['yes', 'no'], default='no', help="Show the model's architecture")
    args = parser.parse_args()
    main(args)
