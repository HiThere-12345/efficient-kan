from kan import KAN
from functions import evaluate, save_kan_model, load_kan_model, data_subset

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


transform = transforms.Compose([
    transforms.ToTensor(),  
    transforms.Normalize((0.5,), (0.5,))
])
full_trainset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
full_valset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

#Subsets of the data
trainset = data_subset(full_trainset, 1)
valset = data_subset(full_valset, 1)
print(f"Using {len(trainset)} training samples out of {len(full_trainset)} total")
print(f"Using {len(valset)} validation samples out of {len(full_valset)} total")

#Data Loaders
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
valloader = DataLoader(valset, batch_size=64, shuffle=False)

#Initialize the model
model = KAN([28 * 28, 64, 10])

#Other Variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
criterion = nn.CrossEntropyLoss()

#Train the model
for epoch in range(10):
    model.train()
    with tqdm(trainloader) as pbar:
        for i, (images, labels) in enumerate(pbar):
            images = images.view(-1, 28 * 28).to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels.to(device))
            loss.backward()
            optimizer.step()
            accuracy = (output.argmax(dim=1) == labels.to(device)).float().mean()
            pbar.set_postfix(loss=loss.item(), accuracy=accuracy.item(), lr=optimizer.param_groups[0]['lr'])

    # Validation
    model.eval()
    val_loss, val_accuracy = evaluate(valloader, model, criterion)
    scheduler.step()
    print(f"Epoch {epoch + 1}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}")

#Save and more evaluation
save_kan_model(model, path="./model/x")
fullloader = DataLoader(full_valset, batch_size=64, shuffle=False)
total_loss, total_accuracy = evaluate(fullloader, model, criterion)
print(f"Total Loss: {total_loss}, Total Accuracy: {total_accuracy}")