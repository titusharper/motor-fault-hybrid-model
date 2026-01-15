import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionFiLMModel(nn.Module):
    """
    - Convolutions: 5x Conv-BN (stride=2). FiLM = conv4 ve conv5 (After BN, before ReLu).
    - 1D Branch: FC(->64)->Dropout->FC(->128).
    - After GAP 256-d -> classifier(64) -> num_classes.
    - backbone: conv+bn layers.
    """
    def __init__(self, num_numeric_features, num_classes, freeze_backbone=False):
        super().__init__()
        # --- CNN layers ---
        self.conv1 = nn.Conv2d(3, 16,  kernel_size=3, stride=2, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32,  kernel_size=3, stride=2, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64,  kernel_size=3, stride=2, padding=1)
        self.bn3   = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn4   = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128,256, kernel_size=3, stride=2, padding=1)
        self.bn5   = nn.BatchNorm2d(256)
        self.dropout2d = nn.Dropout2d(0.2)

        # backbone
        self.backbone = nn.ModuleList([self.conv1,self.bn1,self.conv2,self.bn2,
                                       self.conv3,self.bn3,self.conv4,self.bn4,
                                       self.conv5,self.bn5])

        # --- 1D Branch ---
        self.fc1    = nn.Linear(num_numeric_features, 64)
        self.bn_fc1 = nn.BatchNorm1d(64)
        self.fc2    = nn.Linear(64, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.drop_fc = nn.Dropout(0.2)

        # --- FiLM parameters ---
        self.gamma4_fc = nn.Linear(128, 128)   # conv4 
        self.beta4_fc  = nn.Linear(128, 128)
        self.gamma5_fc = nn.Linear(128, 256)   # conv5 
        self.beta5_fc  = nn.Linear(128, 256)

        # --- Classifier ---
        self.fc_clf1 = nn.Linear(256, 64)
        self.bn_clf  = nn.BatchNorm1d(64)
        self.dropout_clf = nn.Dropout(0.5)
        self.fc_out  = nn.Linear(64, num_classes)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x_img, x_num):
        # First three blocks 
        x = F.relu(self.bn1(self.conv1(x_img)))   # -> (B,16,112,112)
        x = F.relu(self.bn2(self.conv2(x)))       # -> (B,32,56,56)
        x = F.relu(self.bn3(self.conv3(x)))       # -> (B,64,28,28)

        # Calculating 1D Branch
        z = F.relu(self.bn_fc1(self.fc1(x_num)))
        z = self.drop_fc(z)
        z = F.relu(self.bn_fc2(self.fc2(z)))      # -> (B,128)

        # --- 4. block + FiLM(128ch) ---
        x4 = self.bn4(self.conv4(x))              # -> (B,128,14,14)
        gamma4 = self.gamma4_fc(z).unsqueeze(-1).unsqueeze(-1)  # (B,128,1,1)
        beta4  = self.beta4_fc(z).unsqueeze(-1).unsqueeze(-1)   # (B,128,1,1)
        x = F.relu(x4 * gamma4 + beta4)

        # --- 5. block + FiLM(256ch) ---
        x5 = self.bn5(self.conv5(x))              # -> (B,256,7,7)
        gamma5 = self.gamma5_fc(z).unsqueeze(-1).unsqueeze(-1)  # (B,256,1,1)
        beta5  = self.beta5_fc(z).unsqueeze(-1).unsqueeze(-1)   # (B,256,1,1)
        x = F.relu(x5 * gamma5 + beta5)

        # Dropout2d + GAP
        x = self.dropout2d(x)
        x = F.adaptive_avg_pool2d(x, 1)           # -> (B,256,1,1)
        x = x.view(x.size(0), -1)                 # -> (B,256)

        # Classifier
        x = F.relu(self.bn_clf(self.fc_clf1(x)))
        x = self.dropout_clf(x)
        logits = self.fc_out(x)
        return logits
