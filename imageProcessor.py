import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import pickle
from clearml import Task  # Import ClearML

# Initialize ClearML Task
task = Task.init(project_name="Image Processing", task_name="Train SimpleCNN")

class EventDataset(Dataset):
    def __init__(self, combined_dataset, root_dir, transform=None, label_column=None):
        self.data_frame = combined_dataset
        self.root_dir = root_dir
        self.transform = transform
        self.label_column = label_column  # Specify the label column

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # Convert sparse tensors back to dense
        images = [
            torch.tensor(self.data_frame.iloc[idx][f"plane{i}_sparse_tensor"].to_dense())
            # for i in range(3)
            for i in [2]
        ]
        images = torch.stack(images, dim=0)  # Stack tensors along a new dimension

        if self.transform:
            images = [self.transform(img) for img in images]
        images = torch.stack(images, dim=0)

        # Use the specified label column
        label = self.data_frame.iloc[idx][self.label_column] if self.label_column else self.data_frame.iloc[idx, 3]
        return images, label

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 64 * 64, 128)
        self.fc2 = nn.Linear(128, 2)  # Assuming binary classification

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 64 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class AlexCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)  # First convolutional layer
        self.bn1 = nn.BatchNorm2d(64)  # Batch normalization after conv1
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)  # First pooling layer

        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2)  # Second convolutional layer
        self.bn2 = nn.BatchNorm2d(192)  # Batch normalization after conv2
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)  # Second pooling layer

        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1)  # Third convolutional layer
        self.bn3 = nn.BatchNorm2d(384)  # Batch normalization after conv3

        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)  # Fourth convolutional layer
        self.bn4 = nn.BatchNorm2d(256)  # Batch normalization after conv4

        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)  # Fifth convolutional layer
        self.bn5 = nn.BatchNorm2d(256)  # Batch normalization after conv5
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)  # Third pooling layer

        self.fc1 = nn.Linear(256 * 6 * 6, 4096)  # First fully connected layer
        self.fc2 = nn.Linear(4096, 4096)  # Second fully connected layer
        self.fc3 = nn.Linear(4096, 2)  # Output layer (binary classification)

        self.dropout = nn.Dropout(0.5)  # Dropout for regularization

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool3(F.relu(self.bn5(self.conv5(x))))

        x = x.view(-1, 256 * 6 * 6)  # Flatten the tensor for fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def load_pickled_datasets(directory_path):
    dataframes = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.pkl'):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'rb') as f:
                df = pickle.load(f)
                dataframes.append(df)
    combined_dataset = pd.concat(dataframes, ignore_index=True)
    return combined_dataset

def train_model(combined_dataset, root_dir, num_epochs=10, batch_size=16, learning_rate=0.001, label_column=None):
    # Log hyperparameters to ClearML
    task.connect({
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "label_column": label_column,
    })

    transform = transforms.Compose([
        # transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    dataset = EventDataset(
        combined_dataset=combined_dataset, 
        root_dir=root_dir, 
        transform=transform, 
        label_column=label_column
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 10 == 9:
                avg_loss = running_loss / 10
                print(f"[{epoch + 1}, {i + 1}] loss: {avg_loss:.3f}")
                running_loss = 0.0

                # Log loss to ClearML
                task.get_logger().report_scalar("Loss", "Train", iteration=epoch * len(dataloader) + i, value=avg_loss)

    print("Finished Training")
    torch.save(model.state_dict(), "cnn_model.pth")

    # Log the model to ClearML
    task.upload_artifact("model", "cnn_model.pth")

if __name__ == "__main__":
    directory_path = "/workspace/outputDir"
    root_dir = "/workspace/outputDir"
    label_column = "is_cc"  # Replace with the column name for labels

    # Load and combine pickled datasets
    combined_dataset = load_pickled_datasets(directory_path)
    print("Combined dataset loaded successfully!")

    # Train the model
    train_model(combined_dataset, root_dir, label_column=label_column)