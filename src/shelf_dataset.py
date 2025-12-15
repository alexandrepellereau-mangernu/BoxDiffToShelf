import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple

class ShelfDataset(Dataset):
    """Dataset pour l'entraînement du réseau de neurones"""
    
    def __init__(self, samples: List[Tuple[dict, str]], max_boxes: int = 20):
        self.samples = samples
        self.max_boxes = max_boxes
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        data, label = self.samples[idx]
        features = self._extract_features(data)
        label_array = np.array([int(x) for x in label.split('|')])
        return torch.FloatTensor(features), torch.FloatTensor(label_array)
    
    def _extract_features(self, data: dict) -> np.ndarray:
        """Extrait et normalise les features des boxes"""
        features = []
        
        for key in ['before', 'after']:
            boxes = data.get(key, [])
            
            # Normaliser les coordonnées (supposer image 640x480)
            normalized_boxes = []
            for box in boxes[:self.max_boxes]:
                x, y, h, w = box
                normalized_boxes.append([x/640.0, y/480.0, h/100.0, w/100.0])
            
            # Padding
            padded = np.zeros((self.max_boxes, 4))
            if len(normalized_boxes) > 0:
                padded[:len(normalized_boxes)] = normalized_boxes
            
            features.append(padded.flatten())
            # Nombre de boxes normalisé
            features.append(np.array([len(boxes) / 10.0]))
        
        return np.concatenate(features)