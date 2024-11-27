import torch
from torch import nn
from torch.nn import functional as F

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        #We go from 2 to 3 layers and we increase feature map depth.
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1) #otputs 16 channels 
        self.bn1 = nn.BatchNorm2d(16) #for each layer we add Batch Normalization
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # We add Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Multiple fully connected layers
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.fc1_relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, 128)
        self.fc2_relu = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(128, 10)
        
        pass

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with dropout
        x = self.dropout(self.fc1_relu(self.fc1(x)))
        x = self.dropout(self.fc2_relu(self.fc2(x)))
        x = self.fc3(x)
        return x
