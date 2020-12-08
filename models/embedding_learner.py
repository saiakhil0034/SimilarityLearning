import torch.nn as nn
import torch.nn.functional as F


class EmbeddingNet(nn.Module):
    def __init__(self, ni=128, no=64):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv3d(ni, 128, 2, padding=2), nn.PReLU(),
                                     nn.MaxPool3d(2, stride=1),
                                     nn.Conv3d(128, 64, 2), nn.PReLU(),
                                     nn.MaxPool3d(2, stride=1),
                                     nn.Conv3d(64, 32, 2), nn.PReLU(),
                                     nn.MaxPool3d(2, stride=1),
                                     )

        self.fc = nn.Sequential(nn.Linear(32 * 4 * 4 * 4, 256),
                                nn.PReLU(),
                                nn.Linear(256, 128),
                                nn.PReLU(),
                                nn.Linear(128, no)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        with torch.no_grad():
            return self.forward(x)
