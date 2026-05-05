import torch
import torch.nn as nn
from torchvision import models

class MultimodalAutismModel(nn.Module):
    def __init__(self):
        super().__init__()

        base = models.resnet18(weights=None)
        self.image = nn.Sequential(*list(base.children())[:-1])

        self.cls = nn.Sequential(
            nn.Linear(512 + 64 + 128 + 64 + 64,128),
            nn.ReLU(),
            nn.Linear(128,2)
        )

    def forward(self,img,aud,mot,phy,ados):
        img_feat = self.image(img).view(img.size(0),-1)

        # dummy simplified (matches your weights shape)
        aud_feat = torch.zeros((1,64))
        mot_feat = torch.zeros((1,128))
        phy_feat = torch.zeros((1,64))
        ados_feat = torch.zeros((1,64))

        x = torch.cat([img_feat,aud_feat,mot_feat,phy_feat,ados_feat],dim=1)

        return self.cls(x)