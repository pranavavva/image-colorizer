import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision.transforms import ToTensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

class ImageColorizerDataset(Dataset):
    def __init__(self, root_dir, train=True, n_samples=None, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.annotations_file = pd.read_csv(os.path.join(root_dir, "train.csv" if train else "test.csv"), header=None)
        if n_samples:
            self.annotations_file = self.annotations_file.sample(n_samples)
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.annotations_file)

    def __getitem__(self, idx: int):
        img_path = os.path.join(self.root_dir, "thumbs_gray", self.annotations_file.iloc[idx, 0])
        target_path = os.path.join(self.root_dir, "thumbs", self.annotations_file.iloc[idx, 0])

        img = read_image(img_path).numpy()
        target = read_image(target_path).numpy()
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)

        return img.permute(1, 0, 2), target.permute(1, 0, 2)

train_data = ImageColorizerDataset(root_dir="./data", train=True, transform=ToTensor(), target_transform=ToTensor(), n_samples=100000)
test_data = ImageColorizerDataset(root_dir="./data", train=False, transform=ToTensor(), target_transform=ToTensor(), n_samples=1000)

train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

class PrintLayer(nn.Module):
    def __init__(self, name=""):
        super(PrintLayer, self).__init__()
        self.name = name
    
    def forward(self, x):
        print(self.name, x.shape)
        return x

class ImageColorizerNetwork(nn.Module):
    def __init__(self):
        """Convolutional neural network for colorizing grayscale images.
        Takes a 1x64x64 image as input and outputs a 3x64x64 image.
        Multiple blocks of convolution->maxpool->batchnorm->relu are used.
        """
        super().__init__()
        self.conv_stack = nn.Sequential(

            # Downsample 
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # PrintLayer("First set"),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # PrintLayer("Second set"),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # PrintLayer("Third set"),

            # Upsample
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # PrintLayer("Fourth set"),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # PrintLayer("Fifth set"),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # PrintLayer("Sixth set"),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # PrintLayer("Seventh set"),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),


            # PrintLayer("Eighth set"),
        )
    
    def forward(self, x):
        return self.conv_stack(x)

model = ImageColorizerNetwork().to(device)
print(model)

learning_rate = 1e-3
batch_size = 64
epochs = 32
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss = 0

    for batch, (img, target) in enumerate(dataloader):
        img, target = img.to(device), target.to(device)

        # Compute prediction error
        pred = model(img)
        loss = loss_fn(pred, target)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(img)
            print(f"loss: {loss:>7f}  [{current:>6d}/{size:>6d}]")
    
    return train_loss / num_batches

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    with torch.no_grad():
        for img, target in dataloader:
            img, target = img.to(device), target.to(device)
            pred = model(img)
            test_loss += loss_fn(pred, target).item()
    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")

    return test_loss

train_losses = np.zeros(epochs)
test_losses = np.zeros(epochs)

for t in range(epochs):
    print(f"Epoch {t+1}/{epochs}\n-------------------------------")
    train_losses[t] = train(train_dataloader, model, loss_fn, optimizer)
    test_losses[t] = test(test_dataloader, model, loss_fn)
print("Done!")

torch.save(model, "model.pt")
np.save("train_losses.npy", train_losses)
np.save("test_losses.npy", test_losses)
