import torch
import torch.nn as nn
from torchvision import models

class LandClassifierModel(nn.Module):
    def __init__(self, num_classes=2, freeze_features=True):
        super().__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        if freeze_features:
            for param in self.model.parameters():
                param.requires_grad = False
        num_ftrs = self.model.fc.in_features 
        
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)

