import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from autogluon.tabular import TabularPredictor, TabularDataset

def load_data(data_dir):
    train_data = pd.read_csv(f'{data_dir}/train.csv')
    test_data = pd.read_csv(f'{data_dir}/test.csv')
    val_data = pd.read_csv(f'{data_dir}/validation.csv')
    
    # Drop the ID column
    train_data = train_data.drop(columns=['ID'])
    test_data = test_data.drop(columns=['ID'])
    val_data = val_data.drop(columns=['ID'])
    
    # Encode labels
    label_encoder = LabelEncoder()
    train_data['Label'] = label_encoder.fit_transform(train_data['Label'])
    test_data['Label'] = label_encoder.transform(test_data['Label'])
    val_data['Label'] = label_encoder.transform(val_data['Label'])
    
    return train_data, test_data, val_data

def get_model_architecture(predictor, input_feature_size):
    best_model_name = predictor.get_model_best()
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
                layers.append(nn.BatchNorm1d(current_input_size))
            elif layer_type == nn.Linear:
                layers.append(nn.Linear(current_input_size, layer_obj.out_features))
                current_input_size = layer_obj.out_features
            elif layer_type == nn.ReLU:
                layers.append(nn.ReLU())
            elif layer_type == nn.Dropout:
                layers.append(nn.Dropout(p=layer_obj.p))
            elif layer_type == nn.Softmax:
                layers.append(nn.Softmax(dim=layer_obj.dim))
            else:
                raise ValueError(f"Unhandled layer type: {layer_type}")
        self.main_block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.main_block(x)

def train_pytorch_model(model, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, num_epochs=20, batch_size=64):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        permutation = torch.randperm(X_train_tensor.size()[0])
        for i in range(0, X_train_tensor.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_X, batch_y = X_train_tensor[indices], y_train_tensor[indices]
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Validate the model
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            _, val_predicted = torch.max(val_outputs.data, 1)
            val_accuracy = accuracy_score(y_val_tensor, val_predicted)
        model.train()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Validation Accuracy: {val_accuracy:.4f}')
    
    return model

def evaluate_model(model, X_test_tensor, y_test_tensor):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = accuracy_score(y_test_tensor, predicted)
    return accuracy

def main():
    parser = argparse.ArgumentParser(description="Replicate AutoGluon model architecture in PyTorch")
    parser.add_argument('--predictor_path', type=str, required=True, help="Path to the AutoGluon predictor file")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory where train.csv, test.csv, and validation.csv are located")
    args = parser.parse_args()
    
    # Load data
    train_data, test_data, val_data = load_data(args.data_dir)
    
    # Determine the input feature size
    input_feature_size = train_data.drop(columns=['Label']).shape[1]
    
    # Convert to TabularDataset
    train_data_tab = TabularDataset(train_data)
    test_data_tab = TabularDataset(test_data)
    val_data_tab = TabularDataset(val_data)
    
    # Load predictor
    predictor = TabularPredictor.load(args.predictor_path)
    
    # Get model architecture
    architecture, best_model, input_feature_size = get_model_architecture(predictor, input_feature_size)
    
    # Print AutoGluon model architecture
    print("AutoGluon Model Architecture:")
    print(best_model.model)
    
    # Replicate model in PyTorch
    replicated_model = AutoReplicatedNN(architecture, input_feature_size)
    print("\nReplicated PyTorch Model Architecture:")
    print(replicated_model)
    
    # Prepare training and validation data
    X_train = train_data.drop(columns=['Label']).values
    y_train = train_data['Label'].values
    X_val = val_data.drop(columns=['Label']).values
    y_val = val_data['Label'].values

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    
    # Train PyTorch model
    replicated_model = train_pytorch_model(replicated_model, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor)
    
    # Prepare test data
    X_test = test_data.drop(columns=['Label']).values
    y_test = test_data['Label'].values

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    # Evaluate AutoGluon model
    original_ag_model_accuracy = predictor.evaluate(test_data_tab)['accuracy']
    print(f'Accuracy on the test set (Original AutoGluon Model): {original_ag_model_accuracy:.4f}')
    
    # Evaluate replicated PyTorch model
    replicated_accuracy = evaluate_model(replicated_model, X_test_tensor, y_test_tensor)
    print(f'Accuracy on the test set (Replicated PyTorch Model): {replicated_accuracy:.4f}')

if __name__ == "__main__":
    main()
