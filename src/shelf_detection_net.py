import torch.nn as nn

class ShelfDetectionNet(nn.Module):
    """Réseau amélioré avec attention sur les différences"""
    
    def __init__(self, num_shelves: int = 4, max_boxes: int = 20):
        super(ShelfDetectionNet, self).__init__()
        
        input_size = 2 * (max_boxes * 4 + 1)
        
        # Encoder principal
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )
        
        # Tête de régression
        self.regressor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_shelves)
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.regressor(x)
        return x