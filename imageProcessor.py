import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import sqlite3
import numpy as np
from PIL import Image
import io

def fetch_data_from_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT image, label FROM images_table")
    data = cursor.fetchall()
    conn.close()
    return data

class ImageDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_data, label = self.data[idx]
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def create_dataloader(data, resize_dims=(128, 128), batch_size=32, val_split=0.2):
    transform = transforms.Compose([
        transforms.Resize(resize_dims),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = ImageDataset(data, transform=transform)
    
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

class SimpleCNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 1)
        self.fc1 = nn.Linear(256 * input_size[0] * input_size[1], 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))  
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def calculate_conv_output_size(input_size, kernel_size, stride=1, padding=0):
    return ((input_size - kernel_size + 2 * padding) // stride) + 1

def get_conv_output_size(input_size, conv_layers):
    size = input_size
    for layer in conv_layers:
        size = calculate_conv_output_size(size, layer.kernel_size[0], layer.stride[0], layer.padding[0])
        size //= 2  
    return size

def train_model(dataloader, model, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}")

def evaluate_model(dataloader, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {100 * correct / total}%")

if __name__ == "__main__":
    db_path = os.getenv('DB_PATH')
    resize_dims = (128, 128)
    data = fetch_data_from_db(db_path)
    train_loader, val_loader = create_dataloader(data, resize_dims=resize_dims)
    
    conv_layers = [nn.Conv2d(3, 32, 3, 1), nn.Conv2d(32, 64, 3, 1), nn.Conv2d(64, 128, 3, 1), nn.Conv2d(128, 256, 3, 1)]
    conv_output_size = get_conv_output_size(resize_dims[0], conv_layers)
    model = SimpleCNN((conv_output_size, conv_output_size))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_model(train_loader, model, criterion, optimizer)
    evaluate_model(val_loader, model)