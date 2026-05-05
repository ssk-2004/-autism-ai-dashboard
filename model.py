import torch
import torch.nn as nn
from torchvision import models

ADOS_DIM = 12
PHYSIO_FEAT_DIM = 4


class ImageEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        base = models.resnet18(pretrained=False)

        self.backbone = nn.Sequential(
            *list(base.children())[:-1]
        )

    def forward(self, x):

        x = self.backbone(x)

        return x.view(x.size(0), -1)


class AudioEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.ReLU()
        )

    def forward(self, x):

        x = x.unsqueeze(1)

        x = self.conv(x)

        return x.mean(dim=[2,3])


class MotionEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=128,
            batch_first=True
        )

    def forward(self, x):

        x = x.view(x.size(0), -1, 1)

        _, (h, _) = self.lstm(x)

        return h[-1]


class PhysioEncoder(nn.Module):

    def __init__(self, d):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(d * 2, 64),
            nn.ReLU()
        )

    def forward(self, x):

        mean = x.mean(dim=1)

        std = x.std(dim=1)

        return self.fc(torch.cat([mean, std], dim=1))


class ADOSEncoder(nn.Module):

    def __init__(self, d):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(d, 64),
            nn.ReLU()
        )

    def forward(self, x):

        return self.fc(x)


class MultimodalAutismModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.image = ImageEncoder()

        self.audio = AudioEncoder()

        self.motion = MotionEncoder()

        self.physio = PhysioEncoder(PHYSIO_FEAT_DIM)

        self.ados = ADOSEncoder(ADOS_DIM)

        self.cls = nn.Sequential(
            nn.Linear(512 + 64 + 128 + 64 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )


    def forward(self, img, aud, mot, phy, ados):

        features = torch.cat([

            self.image(img),

            self.audio(aud),

            self.motion(mot),

            self.physio(phy),

            self.ados(ados)

        ], dim=1)

        return self.cls(features)
