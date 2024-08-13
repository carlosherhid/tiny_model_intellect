"""
This script replicates an AutoGluon model architecture in PyTorch and applies Quantization-Aware Training (QAT).
It performs the following steps:
1. Loads the dataset and preprocesses it.
2. Loads the best model from an AutoGluon predictor.
3. Extracts the architecture of the AutoGluon model and replicates it in PyTorch.
4. Trains the replicated PyTorch model.
5. Applies QAT to the PyTorch model, trains it, and collects statistics.
6. Converts the QAT-trained model to a quantized model.
7. Evaluates the accuracy of both the normal and QAT PyTorch models on the test dataset.
8. Saves the trained models and prints their sizes.

Usage:
    python autogluon_to_pytorch_qat.py --predictor_path <path_to_autogluon_predictor> --data_dir <data_directory> --epochs <number_of_epochs> --batch_size <batch_size> --output_dir <output_directory> --qat --normal --experiment_number <experiment_number> --modify_test --scale_close_to_zero --disable_early_stopping --disable_lr_scheduler

Arguments:
    --predictor_path: Path to the AutoGluon predictor file.
    --data_dir: Directory where train.csv, test.csv, and validation.csv are located.
    --epochs: Number of maximum training epochs (default is 1000).
    --batch_size: Batch size for training (default is 256).
    --output_dir: Directory to save the PyTorch model.
    --qat: Flag to apply Quantization-Aware Training.
    --normal: Flag to generate a normal (non-quantized) PyTorch model.
    --experiment_number: Experiment number to include in the model name.
    --modify_test: Flag to modify the test dataset by setting values close to 0 to exactly 0.
    --scale_close_to_zero: Flag to scale values close to 0 by a large factor.
    --disable_early_stopping: Flag to disable early stopping during training.
    --disable_lr_scheduler: Flag to disable the learning rate scheduler during training.

Example:
    python autogluon_to_pytorch_qat.py --predictor_path ./predictor.pkl --data_dir ./datasets --epochs 50 --batch_size 128 --output_dir ./output --qat --normal --experiment_number 1 --modify_test --scale_close_to_zero --disable_early_stopping --disable_lr_scheduler
"""

import argparse
import os
import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from autogluon.tabular import TabularPredictor
from torch.utils.data import DataLoader, TensorDataset
from torch.quantization import QuantStub, DeQuantStub, prepare_qat, convert
from tqdm import tqdm

class Logger(object):
    def __init__(self, output_dir, experiment_number):
        self.terminal = sys.stdout
        self.log = open(os.path.join(output_dir, f"execution_log_exp{experiment_number}.txt"), "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def load_data(data_dir, modify_test=False, scale_close_to_zero=False):
    train_data = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test_data = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    val_data = pd.read_csv(os.path.join(data_dir, 'validation.csv'))
    
    # Drop the ID column
    train_data = train_data.drop(columns=['ID'])
    test_data = test_data.drop(columns=['ID'])
    val_data = val_data.drop(columns=['ID'])
    
    # Encode labels
    label_encoder = LabelEncoder()
    train_data['Label'] = label_encoder.fit_transform(train_data['Label'])
    test_data['Label'] = label_encoder.transform(test_data['Label'])
    val_data['Label'] = label_encoder.transform(val_data['Label'])

    if modify_test:
        test_data_z = test_data.copy()
        # Set values close to 0 to 0
        test_data_z.iloc[:, :-1] = test_data_z.iloc[:, :-1].applymap(lambda x: 0 if abs(x) < 1e-6 else x)
        test_data_z.to_csv(os.path.join(data_dir, 'test_z.csv'), index=False)
        test_data = test_data_z

    if scale_close_to_zero:
        scale_factor = 1e6  # Large factor to scale values
        scaled_test_data = test_data.copy()
        scaled_test_data.iloc[:, :-1] = scaled_test_data.iloc[:, :-1].applymap(lambda x: x * scale_factor if abs(x) < 1e-6 else x)
        scaled_test_data.to_csv(os.path.join(data_dir, 'test_scaled.csv'), index=False)
        test_data = scaled_test_data

    return train_data, test_data, val_data

def create_dataloaders(train_data, test_data, val_data, batch_size):
    # Split features and labels
    X_train, y_train = train_data.drop(columns=['Label']).values, train_data['Label'].values
    X_test, y_test = test_data.drop(columns=['Label']).values, test_data['Label'].values
    X_val, y_val = val_data.drop(columns=['Label']).values, val_data['Label'].values

    # Convert to torch tensors
    train_tensor = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_tensor = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    val_tensor = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    
    # Create DataLoader
    train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_tensor, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_tensor, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, val_loader

def get_model_architecture(predictor, input_feature_size):
    best_model_name = predictor._trainer.model_best
    best_model = predictor._trainer.load_model(best_model_name)
    
    architecture = []
    for name, layer in best_model.model.named_children():
        if isinstance(layer, nn.Sequential):
            for sub_layer in layer:
                architecture.append((type(sub_layer), sub_layer))
        else:
            architecture.append((type(layer), layer))
    
    return architecture, best_model, input_feature_size

class AutoReplicatedNN(nn.Module):
    def __init__(self, architecture, input_feature_size):
        super(AutoReplicatedNN, self).__init__()
        layers = []
        current_input_size = input_feature_size
        for layer_type, layer_obj in architecture:
            if layer_type == nn.BatchNorm1d:
                layers.append(nn.Identity())  # Replace BatchNorm1d with Identity
            elif layer_type == nn.Linear:
                layers.append(nn.Linear(current_input_size, layer_obj.out_features))
                current_input_size = layer_obj.out_features
            elif layer_type == nn.ReLU:
                layers.append(nn.ReLU())
            elif layer_type == nn.Dropout:
                layers.append(nn.Dropout(p=layer_obj.p))
            elif layer_type != nn.Softmax:  # Exclude Softmax
                raise ValueError(f"Unhandled layer type: {layer_type}")
        self.main_block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.main_block(x)

class QATWrapper(nn.Module):
    def __init__(self, model):
        super(QATWrapper, self).__init__()
        self.quant = QuantStub()
        self.model = model
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x

def train_pytorch_model(model, train_loader, val_loader, epochs, device, use_early_stopping=True, use_lr_scheduler=True):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5) if use_lr_scheduler else None

    # Introduce Early Stopping Mechanism
    best_val_accuracy = 0
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for data in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', unit='batch'):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")
        
        # Validate the model
        val_loss, val_accuracy = validate_pytorch_model(model, val_loader, device)
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

        # Step the scheduler
        if scheduler:
            scheduler.step(val_loss)

        # Early stopping check
        if use_early_stopping:
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break

    return model

def validate_pytorch_model(model, val_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = running_loss / len(val_loader)
    val_accuracy = correct / total
    return val_loss, val_accuracy

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    softmax = nn.Softmax(dim=1)  # Initialize Softmax

    with torch.no_grad():
        for data in tqdm(test_loader, desc='Testing', unit='batch'):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            outputs = softmax(outputs)  # Apply Softmax
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (KB):', os.path.getsize("temp.p")/1e3)
    os.remove('temp.p')

def save_model(model, output_dir, model_name, experiment_number):
    model_path = os.path.join(output_dir, f"{model_name}_exp{experiment_number}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

def main():
    parser = argparse.ArgumentParser(description='Replicate AutoGluon model in PyTorch and apply QAT')
    parser.add_argument('--predictor_path', type=str, required=True, help='Path to the AutoGluon predictor')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the dataset CSV files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the output models')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train (default: 1000)')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training (default: 256)')
    parser.add_argument('--qat', action='store_true', help='Apply Quantization-Aware Training')
    parser.add_argument('--normal', action='store_true', help='Train a normal (non-quantized) model')
    parser.add_argument('--experiment_number', type=int, required=True, help='Experiment number to include in model name')
    parser.add_argument('--modify_test', action='store_true', help='Modify test dataset by setting values close to 0 to 0')
    parser.add_argument('--scale_close_to_zero', action='store_true', help='Scale values close to 0 by a large factor')
    parser.add_argument('--disable_early_stopping', action='store_true', help='Disable early stopping during training')
    parser.add_argument('--disable_lr_scheduler', action='store_true', help='Disable the learning rate scheduler during training')

    args = parser.parse_args()

    # Setup logger
    sys.stdout = Logger(args.output_dir, args.experiment_number)

    # Load the data
    train_data, test_data, val_data = load_data(args.data_dir, args.modify_test, args.scale_close_to_zero)
    train_loader, test_loader, val_loader = create_dataloaders(train_data, test_data, val_data, args.batch_size)

    # Load the AutoGluon predictor
    predictor = TabularPredictor.load(args.predictor_path)

    # Get the model architecture from the AutoGluon predictor
    architecture, best_model, input_feature_size = get_model_architecture(predictor, len(train_data.columns) - 1)

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.normal:
        # Replicate and train normal PyTorch model
        normal_model = AutoReplicatedNN(architecture, input_feature_size).to(device)
        print("\nTraining Normal PyTorch Model:")
        normal_model = train_pytorch_model(normal_model, train_loader, val_loader, args.epochs, device, not args.disable_early_stopping, not args.disable_lr_scheduler)
        save_model(normal_model, args.output_dir, "normal_pytorch_model", args.experiment_number)
        normal_accuracy = evaluate_model(normal_model, test_loader, device)
        print(f'Accuracy of the normal model on the test dataset: {normal_accuracy * 100:.2f}%')
        print_size_of_model(normal_model)
    
    if args.qat:
        # Apply QAT and train QAT PyTorch model
        qat_model = AutoReplicatedNN(architecture, input_feature_size).to(device)
        
        qat_model = QATWrapper(qat_model).to(device)
        qat_model.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
        prepare_qat(qat_model, inplace=True)

        print("\nTraining QAT PyTorch Model:")
        qat_model = train_pytorch_model(qat_model, train_loader, val_loader, args.epochs, device, not args.disable_early_stopping, not args.disable_lr_scheduler)
        
        print("\nCollecting statistics during training:")
        print(qat_model)

        qat_model.eval()
        qat_model = qat_model.cpu()  # Move to CPU for quantization
        qat_model = convert(qat_model)
        save_model(qat_model, args.output_dir, "qat_pytorch_model", args.experiment_number)

        qat_accuracy = evaluate_model(qat_model, test_loader, torch.device("cpu"))
        print(f'Accuracy of the QAT model on the test dataset: {qat_accuracy * 100:.2f}%')
        print_size_of_model(qat_model)

        print("Weights after quantization:")
        # Print weights from the first Linear layer
        for layer in qat_model.model.main_block:
            if isinstance(layer, nn.Linear):
                print(torch.int_repr(layer.weight))
                break

    # Evaluate AutoGluon model
    original_ag_model_accuracy = predictor.evaluate(test_data)['accuracy']
    print(f'Accuracy on the test set (Original AutoGluon Model): {original_ag_model_accuracy:.4f}')

if __name__ == "__main__":
    main()
