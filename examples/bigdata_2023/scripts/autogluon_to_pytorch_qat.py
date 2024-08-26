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
from torch.nn.utils import prune
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
    # Load original data
    train_data = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test_data = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    val_data = pd.read_csv(os.path.join(data_dir, 'validation.csv'))

    # Encode labels
    label_encoder = LabelEncoder()
    train_data['Label'] = label_encoder.fit_transform(train_data['Label'])
    test_data['Label'] = label_encoder.transform(test_data['Label'])
    val_data['Label'] = label_encoder.transform(val_data['Label'])

    modified_data_suffix = ""

    def modify_dataset(dataset, operation):
        # Perform operations like modifying or scaling
        modified_dataset = dataset.copy()
        original_types = dataset.dtypes
        modified_dataset.iloc[:, :-1] = modified_dataset.iloc[:, :-1].applymap(operation)
        # Ensure the original types are preserved
        for col, dtype in original_types.items():
            modified_dataset[col] = modified_dataset[col].astype(dtype)
        return modified_dataset

    if modify_test:
        train_data = modify_dataset(train_data, lambda x: 0 if abs(x) < 1e-6 else x)
        test_data = modify_dataset(test_data, lambda x: 0 if abs(x) < 1e-6 else x)
        val_data = modify_dataset(val_data, lambda x: 0 if abs(x) < 1e-6 else x)
        modified_data_suffix += "_modified"

    if scale_close_to_zero:
        scale_factor = 1e6
        train_data = modify_dataset(train_data, lambda x: x * scale_factor if abs(x) < 1e-6 else x)
        test_data = modify_dataset(test_data, lambda x: x * scale_factor if abs(x) < 1e-6 else x)
        val_data = modify_dataset(val_data, lambda x: x * scale_factor if abs(x) < 1e-6 else x)
        modified_data_suffix += "_scaled"

    # Save modified datasets to new CSV files
    if modified_data_suffix:
        train_data.to_csv(os.path.join(data_dir, f'train{modified_data_suffix}.csv'), index=False)
        test_data.to_csv(os.path.join(data_dir, f'test{modified_data_suffix}.csv'), index=False)
        val_data.to_csv(os.path.join(data_dir, f'validation{modified_data_suffix}.csv'), index=False)

    return train_data, test_data, val_data

def prune_features(train_data, test_data, val_data, important_features_file):
    # Read the important features from the file, assuming no header
    pruned_features = pd.read_csv(important_features_file, header=None).squeeze().tolist()
    important_features = ['ID'] + pruned_features + ['Label']

    # Set unimportant features to zero instead of removing them
    def set_unimportant_features_to_zero(dataset):
        unimportant_features = [col for col in dataset.columns if col not in important_features]
        dataset[unimportant_features] = 0
        return dataset

    train_data_pruned = set_unimportant_features_to_zero(train_data)
    test_data_pruned = set_unimportant_features_to_zero(test_data)
    val_data_pruned = set_unimportant_features_to_zero(val_data)

    return train_data_pruned, test_data_pruned, val_data_pruned

def create_dataloaders(train_data, test_data, val_data, batch_size):
    # Split features and labels
    X_train, y_train = train_data.drop(columns=['Label', 'ID']).values, train_data['Label'].values
    X_test, y_test = test_data.drop(columns=['Label', 'ID']).values, test_data['Label'].values
    X_val, y_val = val_data.drop(columns=['Label', 'ID']).values, val_data['Label'].values

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
            if layer_type == nn.Linear:
                layers.append(nn.Linear(current_input_size, layer_obj.out_features))
                current_input_size = layer_obj.out_features
            elif layer_type == nn.ReLU:
                layers.append(nn.ReLU())
            elif layer_type == nn.Dropout:
                layers.append(nn.Dropout(p=layer_obj.p))
            elif layer_type == nn.BatchNorm1d:
                layers.append(nn.Identity())  # Replace BatchNorm1d with Identity
            elif layer_type == nn.Identity or layer_type == nn.Sequential:
                layers.append(layer_obj)
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

def prune_model(model, amount=0.2):
    """
    Prune 20% of the least important weights in the model.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')  # To make the pruning permanent
    return model

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
    parser.add_argument('--prune_features', action='store_true', help='Prune features based on importance')
    parser.add_argument('--important_features_file', type=str, help='File containing important features to keep after pruning')
    parser.add_argument('--prune_model', action='store_true', help='Prune the PyTorch model after quantization')

    args = parser.parse_args()

    # Setup logger
    sys.stdout = Logger(args.output_dir, args.experiment_number)

    # Load the data
    train_data, test_data, val_data = load_data(
        args.data_dir,
        args.modify_test,
        args.scale_close_to_zero
    )

    # Apply feature pruning if the flag is set
    if args.prune_features:
        train_data, test_data, val_data = prune_features(
            train_data,
            test_data,
            val_data,
            args.important_features_file
        )
        # Save pruned datasets to new CSV files
        train_data.to_csv(os.path.join(args.data_dir, 'train_pruned.csv'), index=False)
        test_data.to_csv(os.path.join(args.data_dir, 'test_pruned.csv'), index=False)
        val_data.to_csv(os.path.join(args.data_dir, 'validation_pruned.csv'), index=False)

    train_loader, test_loader, val_loader = create_dataloaders(train_data, test_data, val_data, args.batch_size)

    # Load the AutoGluon predictor
    predictor = TabularPredictor.load(args.predictor_path)

    # Get the model architecture from the AutoGluon predictor
    architecture, best_model, input_feature_size = get_model_architecture(predictor, len(train_data.columns) - 2)  # Exclude ID and Label

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
        
        # Prune the model if the flag is set
        if args.prune_model:
            qat_model = prune_model(qat_model)

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

    # Evaluate AutoGluon model on the modified test set if any modifications were applied
    if args.modify_test or args.scale_close_to_zero or args.prune_features:
        print("Evaluating AutoGluon model on the modified test dataset.")
        ag_test_data = test_data.drop(columns=['Label'])
        ag_labels = test_data['Label']
        original_ag_model_accuracy = predictor.evaluate(pd.concat([ag_test_data, ag_labels], axis=1))['accuracy']
    else:
        original_ag_model_accuracy = predictor.evaluate(pd.read_csv(os.path.join(args.data_dir, 'test.csv')))['accuracy']
    print(f'Accuracy on the test set (Original AutoGluon Model): {original_ag_model_accuracy:.4f}')

if __name__ == "__main__":
    main()
