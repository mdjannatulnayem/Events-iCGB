
# from sklearnex import patch_sklearn
# patch_sklearn()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch, torch.nn as nn ,torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import os
import pickle
import time


# Define the neural network class
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


def train():

    try:
        # Create the directory if it doesn't exist
        model_dir = './Model/'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Load the dataset
        data = pd.read_csv('./Dataset/spaceWeatherSeverityMulticlass.csv')
        data = data.fillna(0)

        # Features and labels
        X = data[['year', 'month', 'bx_gsm', 'by_gsm', 'bz_gsm', 'bt', 'intensity', 'declination', 'inclination', 'north', 'east', 'vertical', 'horizontal']].values
        y = data['class'].values

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Save the scaler for future use
        with open(os.path.join(model_dir, 'spaceWeatherScaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
        print("Scaler saved successfully!")

        # Convert to PyTorch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.long)

        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        # Initialize the model
        model = SpaceWeatherNet()

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Training loop with timing
        num_epochs = 100
        start_time = time.time()  # Start the timer
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # Print the loss for this epoch
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}')
        end_time = time.time()  # End the timer

        # Calculate and print the elapsed time
        elapsed_time = end_time - start_time
        minutes, seconds = divmod(elapsed_time, 60)
        print(f"\nTraining completed in {int(minutes)} minutes and {int(seconds)} seconds.")

        # Evaluate the model
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Accuracy on test set: {100 * correct / total}%')

        # Save the model
        torch.save(model.state_dict(), os.path.join(model_dir, 'spaceWeatherSeverityModel.pth'))
        print("Model saved successfully!")

    except Exception as e:
        print(f"An error occurred: {str(e)}")



if __name__ == '__main__':
    train()
