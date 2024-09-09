import pickle, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F


# Define the same neural network architecture used during training
class SpaceWeatherNet(nn.Module):
    def __init__(self):
        super(SpaceWeatherNet, self).__init__()
        self.fc1 = nn.Linear(13, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 4)  # 4 output classes

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # No activation here, as we'll use CrossEntropyLoss
        return x


def load_model_and_scaler(model_path, scaler_path):
    # Load the trained model
    model = SpaceWeatherNet()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()  # Set model to evaluation mode
    
    # Load the scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    return model, scaler


def preprocess_data(input_data, scaler):
    # Scale the input data
    scaled_data = scaler.transform(np.array([input_data]))
    return torch.tensor(scaled_data, dtype=torch.float32)


def predict_class(model, preprocessed_data):
    # Perform inference
    with torch.no_grad():
        output = model(preprocessed_data)
        _, predicted_class = torch.max(output, 1)
    return predicted_class.item()


def inference_pipeline(input_data):
    # Paths to the saved model and scaler
    model_path = './Model/spaceWeatherSeverityModel.pth'
    scaler_path = './Model/spaceWeatherScaler.pkl'
    
    # Load the model and scaler
    model, scaler = load_model_and_scaler(model_path, scaler_path)
    
    # Preprocess the input data
    preprocessed_data = preprocess_data(input_data, scaler)
    
    # Make prediction
    predicted_class = predict_class(model, preprocessed_data)
    
    return predicted_class



if __name__ == '__main__':
    # Example input data (features: year, month, bx_gsm, by_gsm, bz_gsm, bt, intensity, declination, inclination, north, east, vertical, horizontal)
    input_data = [2017, 6, 3.2, -1.1, 0.7, 4.5, 8.2, 3.6, 1.9, 0.3, -0.5, 0.7, 1.1]
    
    # Run inference pipeline
    predicted_class = inference_pipeline(input_data)
    
    print(f'Predicted Space Weather Severity Class: {predicted_class}')
