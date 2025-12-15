import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import List, Tuple, Dict

from shelf_dataset import ShelfDataset
from shelf_detection_net import ShelfDetectionNet

class ShelfDetector:
    """Classe principale pour l'entraînement et la prédiction"""
    
    def __init__(self, num_shelves: int = 4, max_boxes: int = 20):
        self.num_shelves = num_shelves
        self.max_boxes = max_boxes
        self.model = ShelfDetectionNet(num_shelves, max_boxes)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def train(self, train_data: List[Tuple[dict, str]], 
              epochs: int = 100, 
              batch_size: int = 32,
              lr: float = 0.001,
              validation_data: List[Tuple[dict, str]] = None):
        """Entraîne le modèle avec validation optionnelle"""
        dataset = ShelfDataset(train_data, max_boxes=self.max_boxes, num_shelves=self.num_shelves)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        if validation_data:
            val_dataset = ShelfDataset(validation_data, max_boxes=self.max_boxes, num_shelves=self.num_shelves)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            total_loss = 0
            
            for features, labels in dataloader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            
            # Validation
            if validation_data:
                val_loss = self._validate(val_dataloader, criterion)
                scheduler.step(val_loss)
                
                if (epoch + 1) % 10 == 0:
                    print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}')
                
                # Early stopping
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= 20:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                if (epoch + 1) % 10 == 0:
                    print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
    
    def _validate(self, dataloader, criterion):
        """Validation du modèle"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for features, labels in dataloader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def predict(self, input_data: dict) -> str:
        """Prédit la rangée modifiée et le nombre d'objets pris"""
        self.model.eval()
        
        dataset = ShelfDataset([(input_data, "0|0|0|0")], max_boxes=self.max_boxes, num_shelves=self.num_shelves)
        features, _ = dataset[0]
        features = features.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(features)
            predictions = output.cpu().numpy()[0]
        
        # Arrondir et limiter les valeurs
        predictions = np.clip(np.round(predictions), -20, 0).astype(int)
        
        return '|'.join(map(str, predictions))
    
    def evaluate(self, test_data: List[Tuple[dict, str]]) -> Dict:
        """Évalue le modèle sur des données de test"""
        dataset = ShelfDataset(test_data, max_boxes=self.max_boxes, num_shelves=self.num_shelves)
        dataloader = DataLoader(dataset, batch_size=32)
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in dataloader:
                features = features.to(self.device)
                outputs = self.model(features)
                predictions = np.clip(np.round(outputs.cpu().numpy()), -20, 0)
                
                all_predictions.extend(predictions)
                all_labels.extend(labels.numpy())
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # Métriques
        mae = np.mean(np.abs(all_predictions - all_labels))
        mse = np.mean((all_predictions - all_labels) ** 2)
        
        # Précision (rangée correcte)
        correct_shelf = np.sum(np.argmin(all_predictions, axis=1) == np.argmin(all_labels, axis=1))
        shelf_accuracy = correct_shelf / len(all_labels)
        
        # Précision exacte (nombre exact d'objets)
        exact_match = np.sum(np.all(all_predictions == all_labels, axis=1))
        exact_accuracy = exact_match / len(all_labels)
        
        return {
            'mae': mae,
            'mse': mse,
            'shelf_accuracy': shelf_accuracy,
            'exact_accuracy': exact_accuracy
        }
    
    def save(self, path: str):
        """Sauvegarde le modèle"""
        torch.save({
            'model_state': self.model.state_dict(),
            'num_shelves': self.num_shelves,
            'max_boxes': self.max_boxes
        }, path)
        print(f"Modèle sauvegardé: {path}")
    
    def load(self, path: str):
        """Charge le modèle"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.num_shelves = checkpoint['num_shelves']
        self.max_boxes = checkpoint['max_boxes']
        print(f"Modèle chargé: {path}")