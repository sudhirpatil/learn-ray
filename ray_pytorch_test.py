import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

def get_dataset():
    return datasets.FashionMNIST(
        root="./data/fashion_mnist",
        train=True,
        download=True,
        transform=ToTensor(),
    )

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Flattens input images from (batch_size, 1, 28, 28) to (batch_size, 784)
        self.flatten = nn.Flatten()
        # A stack of linear layers with ReLU activations in between
        #Goes from raw pixels → basic features → complex features → final prediction
        self.linear_relu_stack = nn.Sequential(
            # Takes a 28×28 image (784 pixels), converts photo into 512 different "descriptions" or features
            nn.Linear(28 * 28, 512),
            # ReLU keeps only positive values, discarding negatives (helps with gradient flow)  # First fully connected layer
            nn.ReLU(),  
            #Creates 512 new, more complex features              
            nn.Linear(512, 512),      # Second fully connected layer
            nn.ReLU(),    
            #Produces 10 scores (one for each digit 0-9), The highest score indicates the predicted digit            # Activation
            nn.Linear(512, 10),       # Output layer (10 classes)
        )

    def forward(self, inputs):
        # Flatten the image to a vector
        inputs = self.flatten(inputs)
        # Pass through the linear layers and activations
        logits = self.linear_relu_stack(inputs)
        return logits

def train_func():
    num_epochs = 3          # Number of passes over the dataset
    batch_size = 64         # Number of samples per mini-batch

    # Load training dataset (FashionMNIST) and create DataLoader for batching
    dataset = get_dataset()
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # Instantiate the neural network model
    model = NeuralNetwork()

    # Define the loss function (for classification)
    criterion = nn.CrossEntropyLoss()
    # Use stochastic gradient descent optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            optimizer.zero_grad()         # Clear gradients from previous batch
            pred = model(inputs)          # Forward pass: make predictions
            loss = criterion(pred, labels) # Compute the loss
            loss.backward()               # Backpropagate error
            optimizer.step()              # Update weights
        # Print loss after each epoch (shows last batch's loss)
        print(f"epoch: {epoch}, loss: {loss.item()}")

# train_func()

#Ray Train
import ray.train.torch
#ray.train.torch.prepare_model and ray.train.torch.prepare_data_loader utility functions to set up your model and data for distributed training. This automatically wraps the model with DistributedDataParallel and places it on the right device, and adds DistributedSampler to the DataLoaders.
def train_func_distributed():
    num_epochs = 3
    batch_size = 64

    dataset = get_dataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataloader = ray.train.torch.prepare_data_loader(dataloader)

    model = NeuralNetwork()
    model = ray.train.torch.prepare_model(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(num_epochs):
        if ray.train.get_context().get_world_size() > 1:
            dataloader.sampler.set_epoch(epoch)

        for inputs, labels in dataloader:
            optimizer.zero_grad()
            pred = model(inputs)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()
        print(f"epoch: {epoch}, loss: {loss.item()}")

#Run the training with 4 distributed workers using Ray
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
# For GPU Training, set `use_gpu` to True.
use_gpu = False
trainer = TorchTrainer(
    train_func_distributed,
    scaling_config=ScalingConfig(num_workers=4, use_gpu=use_gpu)
)

results = trainer.fit()