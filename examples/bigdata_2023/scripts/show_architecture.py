import os
import argparse
import torch
from autogluon.tabular import TabularPredictor

def load_model(model_directory, model_name):
    model_path = os.path.join(model_directory, model_name)
    model_name_predictor = os.path.join(*model_directory.split('/')[-2:])
    predictor_path = os.path.abspath(os.path.join(model_directory, "../../../"))
    
    if model_name.endswith('.pkl'):
        predictor = TabularPredictor.load(predictor_path)
        neural_net_model = predictor._trainer.load_model(model_name_predictor).model
    elif model_name.endswith('.pth'):
        neural_net_model = torch.load(model_path)
    else:
        raise ValueError("Unsupported model file format. Please provide a .pkl or .pth file.")
    
    return neural_net_model

def main(args):
    model_directory = args.model_directory
    model_name = args.model_name
    model = load_model(model_directory, model_name)
    print(f"Model architecture:\n{model}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show the architecture of a given model")
    parser.add_argument('--model-directory', type=str, required=True, help="Path to the directory where the model file is stored")
    parser.add_argument('--model-name', type=str, required=True, help="Name of the model file (.pkl or .pth)")
    
    args = parser.parse_args()
    main(args)
