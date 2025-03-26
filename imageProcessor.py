import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

class EventDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_names = [os.path.join(self.root_dir, self.data_frame.iloc[idx, i]) for i in range(6, 9)]
        images = [Image.open(img_name) for img_name in img_names]
        if self.transform:
            images = [self.transform(img) for img in images]
        images = torch.stack(images, dim=0)
        label = self.data_frame.iloc[idx, 3]  # Assuming 'is_cc' is the label
        return images, label

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
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

def train_model(csv_file, root_dir, num_epochs=10, batch_size=16, learning_rate=0.001):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    dataset = EventDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)
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
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 10:.3f}")
                running_loss = 0.0

    print("Finished Training")
    torch.save(model.state_dict(), "cnn_model.pth")

if __name__ == "__main__":
    csv_file = "/workspace/outputDir/parsed_data_file.csv"  # Update with your actual CSV file path
    root_dir = "/workspace/outputDir"
    train_model(csv_file, root_dir)