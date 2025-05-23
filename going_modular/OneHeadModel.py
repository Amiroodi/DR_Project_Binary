import torch
import torchvision
from torch import nn

def reg_classify(x, device):
    bins = torch.tensor([0.5, 1.5, 2.5, 3.5]).to(device)  # Class boundaries
    # Classify using bucketize
    classified = torch.bucketize(x, bins, right=False)  # right=False ensures correct bin placement
    return classified

class OneHeadModel(nn.Module):
    def __init__(self, device, p_dropout):
        super(OneHeadModel, self).__init__()

        self.device = device
        self.p_dropout = p_dropout

        # Load EfficientNet encoder
        weights = torchvision.models.EfficientNet_B4_Weights.DEFAULT
        efficientNet = torchvision.models.efficientnet_b4(weights=weights)
        self.encoder = efficientNet.features

        # Pooling layers
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Fully connected layers
        self.batch_norm_1= nn.BatchNorm1d(1792) 
        self.batch_norm_2= nn.BatchNorm1d(1792)

        self.dense1 = nn.Linear(1792 * 2, 512)

        # Classification head
        self.classification_head = nn.Sequential(
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Dropout(p=self.p_dropout),
            nn.Linear(32, 1) # 5 output nodes for classification
            )
        
        # Apply He initialization to classification_head
        self._initialize_weights()
        
    def _initialize_weights(self):
        
        # Initialize dense1
        nn.init.kaiming_normal_(self.dense1.weight, mode='fan_in', nonlinearity='relu')
        if self.dense1.bias is not None:
            nn.init.zeros_(self.dense1.bias)

        for module in self.classification_head:
            if isinstance(module, nn.Linear):
                # Apply He initialization to weights
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                # Initialize biases to zero (optional, common practice)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.encoder(x) # Extract features

        # Apply pooling layers
        max_pooled = self.global_max_pool(x).view(x.size(0), -1)
        avg_pooled = self.global_avg_pool(x).view(x.size(0), -1)

        # Concatenate
        x1 = self.batch_norm_1(max_pooled)
        x2 = self.batch_norm_2(avg_pooled)
        x = torch.concat([x1, x2], dim=1)
        x = torch.relu(self.dense1(x))

        # enc_out for visualizing data with t-SNE
        enc_out = x

        # Classification branch
        class_out = self.classification_head(x).squeeze(dim=1)

        return class_out, enc_out

    