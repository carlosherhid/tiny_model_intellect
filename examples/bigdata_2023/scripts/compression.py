import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.quantization import quantize_dynamic
from sklearn.metrics import accuracy_score
from autogluon.tabular import TabularPredictor

class QuantizedModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(QuantizedModelWrapper, self).__init__()
        self.model = model
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x

# Function to evaluate model
def evaluate_model(predictor, X, y):
    y_pred = predictor.predict(X)
    accuracy = accuracy_score(y, y_pred)
    return accuracy

# Function to evaluate quantized model
def evaluate_quantized_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_corrects.double() / total

    print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    return epoch_acc.item()

# Dynamic quantization
def quantize_model_dynamic(model):
    model_quantized = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    return QuantizedModelWrapper(model_quantized)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    model_directory = args.model_directory
    model_name = os.path.join(*model_directory.split('/')[-2:])
    predictor_path = os.path.abspath(os.path.join(model_directory, "../../../"))
    compressed_model_path = os.path.abspath(os.path.join(args.output_directory, "compressed_models"))
    os.makedirs(compressed_model_path, exist_ok=True)

    # Load original model
    original_predictor = TabularPredictor.load(predictor_path)

    # Prepare data
    test_data_path = args.test_data
    test_data = pd.read_csv(test_data_path, index_col="ID")
    X_test = test_data.drop(columns=['Label'])
    y_test = test_data['Label']

    # Evaluate original model
    original_accuracy = evaluate_model(original_predictor, X_test, y_test)
    print(f"Original Model Accuracy: {original_accuracy:.4f}")

    # Load the specific neural network model
    neural_net_model = original_predictor._trainer.load_model(model_name).model.to(device)

    # Display model characteristics if requested
    if args.architecture == 'yes':
        print(f"Original Model architecture:\n{neural_net_model}\n")
        print(f"Original Model state dict keys:\n{neural_net_model.state_dict().keys()}\n")

    # Quantization Dynamic
    print(f"**********QUANTIZATION DYNAMIC**************")
    quantized_dynamic_model = quantize_model_dynamic(neural_net_model)
    quantized_dynamic_model_path = os.path.join(compressed_model_path, "quantized_dynamic_model.pth")
    torch.save(quantized_dynamic_model.state_dict(), quantized_dynamic_model_path)
    print(f"Quantized Dynamic model saved at: {quantized_dynamic_model_path}\n")

    # Model sizes
    original_model_size = os.path.getsize(os.path.join(model_directory, "model.pkl"))
    quantized_dynamic_model_size = os.path.getsize(quantized_dynamic_model_path)

    print(f"Original Model Size: {original_model_size / 1024:.2f} KB")
    print(f"Quantized Dynamic Model Size: {quantized_dynamic_model_size / 1024:.2f} KB")

    # Load the quantized model
    quantized_model = QuantizedModelWrapper(neural_net_model)
    quantized_model.load_state_dict(torch.load(quantized_dynamic_model_path))
    quantized_model.to(device)

    # Display quantized model characteristics if requested
    if args.architecture == 'yes':
        print(f"Quantized Model architecture:\n{quantized_model}")
        print(f"Quantized Model state dict keys:\n{quantized_model.state_dict().keys()}")

    # Prepare test data tensor
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long).to(device)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Evaluate the quantized model
    criterion = torch.nn.CrossEntropyLoss()
    quantized_accuracy = evaluate_quantized_model(quantized_model, test_dataloader, criterion, device)
    print(f"Quantized Model Accuracy: {quantized_accuracy:.4f}")

    # Plot results
    labels = ['Original', 'Quantized Dynamic']
    sizes = [original_model_size, quantized_dynamic_model_size]
    accuracies = [original_accuracy, quantized_accuracy]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    color = 'tab:blue'
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Size (KB)', color=color)
    ax1.bar(labels, [size / 1024 for size in sizes], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    color = 'tab:green'
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Accuracy', color=color)
    ax2.bar(labels, accuracies, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('Model Compression and Performance Comparison')
    plt.savefig(os.path.join(args.output_directory, "compression_and_performance_comparison.png"))
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compress and evaluate AutoGluon models")
    parser.add_argument('--model-directory', type=str, required=True, help="Directory of the best model (e.g., ./datasets/CICIDS2017/balanced_binary/automl_search/models/NeuralNetTorch/a7a36c5f)")
    parser.add_argument('--output-directory', type=str, required=True, help="Directory to save compressed models and results")
    parser.add_argument('--test-data', type=str, required=True, help="Path to the test data CSV file (e.g., ./datasets/CICIDS2017/balanced_binary/test.csv)")
    parser.add_argument('--architecture', type=str, choices=['yes', 'no'], default='no', help="Show the model's architecture")
    args = parser.parse_args()
    main(args)
