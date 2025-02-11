import torch
import torch.nn as nn
from torchvision import models
from utils import make_rotation_matrix
from absl import flags
FLAGS = flags.FLAGS


class SimpleModel(nn.Module):
    def __init__(self, num_classes):
        super(SimpleModel, self).__init__()
        self.resnet = models.resnet18(weights='IMAGENET1K_V1')
        self.resnet.fc = nn.Identity()
        self.num_classes = num_classes
        num_outputs = self.num_classes + 9 + 3
        self.one = 1

        self.head = nn.Sequential(
            nn.Linear(512 + 4, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_outputs)
        )

    def forward(self, image, bbox):
        x = self.resnet(image)
        x = torch.cat((x, bbox), dim=1)
        x = self.head(x)
        logits, R, t = torch.split(x, [self.num_classes, 9*self.one, 3*self.one], dim=1)
        return logits, R, t

    def process_output(self, outs):
        with torch.no_grad():
            logits, R, t = outs 
            cls = logits.argmax(dim=1)
            R = make_rotation_matrix(R.reshape(-1, 3, 3))
            t = t.reshape(-1, 3, 1)
            return cls, R, t
    
