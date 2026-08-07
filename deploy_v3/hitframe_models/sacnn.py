"""
SA-CNN (Shot Angle CNN): Classifies frames as shot angle 0 or 1.
Used to segment rallies from raw video by detecting camera angle changes.
Source: arthur900530/Automated-Hit-frame-Detection-for-Badminton-Match-Analysis
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


class SACNN(nn.Module):
    """Shot Angle CNN - classifies frame as angle 0 (between rallies) or 1 (during rally)."""
    def __init__(self):
        super(SACNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn3 = nn.BatchNorm2d(32)
        self.l1 = nn.Linear(27 * 27 * 32, 2)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.bn1(self.pool1(self.conv1(x)))
        x = F.relu(x)
        x = self.bn2(self.pool2(self.conv2(x)))
        x = F.relu(x)
        x = self.bn3(self.pool3(self.conv3(x)))
        x = F.relu(x)
        x = x.view(-1, 27 * 27 * 32)
        x = self.l1(x)
        x = F.relu(x)
        out = self.dropout(x)
        return out


class SACNNContainer(object):
    """Container for SA-CNN model with preprocessing."""
    def __init__(self, weight_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SACNN().to(self.device)
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.model.eval()
        
        self.preprocess = transforms.Compose([
            transforms.Resize((216, 384)),
            transforms.CenterCrop((216, 216)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def predict(self, img):
        """Predict shot angle: 0=between rallies, 1=during rally."""
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        out = self.model(tensor)
        predicted = torch.argmax(out, dim=1)
        return predicted.item()
