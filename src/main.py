import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict

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
        
        for key in ['left_before', 'right_before', 'left_after', 'right_after']:
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


class ShelfDetectionNet(nn.Module):
    """Réseau amélioré avec attention sur les différences"""
    
    def __init__(self, num_shelves: int = 4, max_boxes: int = 20):
        super(ShelfDetectionNet, self).__init__()
        
        input_size = 4 * (max_boxes * 4 + 1)
        
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
        dataset = ShelfDataset(train_data, max_boxes=self.max_boxes)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        if validation_data:
            val_dataset = ShelfDataset(validation_data, max_boxes=self.max_boxes)
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
        
        dataset = ShelfDataset([(input_data, "0|0|0|0")], max_boxes=self.max_boxes)
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
        dataset = ShelfDataset(test_data, max_boxes=self.max_boxes)
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


# Générateur de données amélioré
def generate_realistic_data(n_samples: int = 500) -> Tuple[List, List]:
    """Génère des données synthétiques réalistes"""
    train_samples = []
    val_samples = []
    
    # Positions fixes des rangées (Y)
    shelf_positions = [50, 150, 250, 350]
    
    for i in range(n_samples):
        # Nombre d'objets initial par rangée (consistant)
        items_per_shelf = [
            np.random.randint(4, 8),
            np.random.randint(4, 8),
            np.random.randint(4, 8),
            np.random.randint(4, 8)
        ]
        
        # Choisir UNE rangée à modifier
        modified_shelf = np.random.randint(0, 4)
        n_removed = np.random.randint(1, min(4, items_per_shelf[modified_shelf]))
        
        # Générer boxes AVANT
        left_before = []
        right_before = []
        
        for shelf_idx, (y_pos, n_items) in enumerate(zip(shelf_positions, items_per_shelf)):
            for j in range(n_items):
                x = 80 + j * 70 + np.random.randint(-5, 5)
                box_h = 40 + np.random.randint(-5, 5)
                box_w = 50 + np.random.randint(-5, 5)
                
                left_before.append([x, y_pos, box_h, box_w])
                right_before.append([x + 320, y_pos, box_h, box_w])
        
        # Générer boxes APRÈS (retirer exactement n_removed de la rangée modifiée)
        left_after = []
        right_after = []
        removed_count = 0
        
        for box in left_before:
            y = box[1]
            shelf_idx = min(range(4), key=lambda i: abs(y - shelf_positions[i]))
            
            if shelf_idx == modified_shelf and removed_count < n_removed:
                removed_count += 1
                continue
            left_after.append(box)
        
        removed_count = 0
        for box in right_before:
            y = box[1]
            shelf_idx = min(range(4), key=lambda i: abs(y - shelf_positions[i]))
            
            if shelf_idx == modified_shelf and removed_count < n_removed:
                removed_count += 1
                continue
            right_after.append(box)
        
        # Label
        label = ['0'] * 4
        label[modified_shelf] = str(-n_removed)
        
        sample = ({
            'left_before': left_before,
            'right_before': right_before,
            'left_after': left_after,
            'right_after': right_after
        }, '|'.join(label))
        
        # Split train/val 80/20
        if i < n_samples * 0.8:
            train_samples.append(sample)
        else:
            val_samples.append(sample)
    
    return train_samples, val_samples


# Exemple d'utilisation
if __name__ == "__main__":
    print("=== Génération des données ===")
    train_data, val_data = generate_realistic_data(1000)
    print(f"Train: {len(train_data)}, Validation: {len(val_data)}")
    
    print("\n=== Entraînement du modèle ===")
    detector = ShelfDetector(num_shelves=4, max_boxes=30)
    detector.train(train_data, validation_data=val_data, epochs=100, batch_size=32, lr=0.001)
    
    print("\n=== Évaluation ===")
    metrics = detector.evaluate(val_data)
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"Précision rangée: {metrics['shelf_accuracy']*100:.2f}%")
    print(f"Précision exacte: {metrics['exact_accuracy']*100:.2f}%")
    
    print("\n=== Test de prédiction ===")
    test_input = {
        'left_before': [[80, 50, 40, 50], [150, 50, 40, 50], [220, 50, 40, 50]],
        'right_before': [[400, 50, 40, 50], [470, 50, 40, 50], [540, 50, 40, 50]],
        'left_after': [[80, 50, 40, 50], [220, 50, 40, 50]],
        'right_after': [[400, 50, 40, 50], [540, 50, 40, 50]]
    }
    
    prediction = detector.predict(test_input)
    print(f"Prédiction: {prediction}")
    print("Attendu: -2|0|0|0 (2 objets retirés de la rangée 0)")
    
    detector.save('shelf_detector.pth')