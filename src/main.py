import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple

class ShelfDataset(Dataset):
    """Dataset pour l'entraînement du réseau de neurones"""
    
    def __init__(self, samples: List[Tuple[dict, str]], max_boxes: int = 20):
        """
        samples: Liste de tuples (input_data, label)
        input_data: dict avec 'left_before', 'right_before', 'left_after', 'right_after'
        label: str format "0|-1|0|0" indiquant les objets pris par rangée
        max_boxes: nombre maximum de boxes par vue
        """
        self.samples = samples
        self.max_boxes = max_boxes
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        data, label = self.samples[idx]
        
        # Convertir les boxes en features
        features = self._extract_features(data)
        
        # Convertir le label en tensor
        label_array = np.array([int(x) for x in label.split('|')])
        
        return torch.FloatTensor(features), torch.FloatTensor(label_array)
    
    def _extract_features(self, data: dict) -> np.ndarray:
        """Extrait les features des boxes"""
        features = []
        
        for key in ['left_before', 'right_before', 'left_after', 'right_after']:
            boxes = data.get(key, [])
            
            # Padding/truncation pour avoir une taille fixe
            padded = np.zeros((self.max_boxes, 4))
            n_boxes = min(len(boxes), self.max_boxes)
            
            if n_boxes > 0:
                padded[:n_boxes] = np.array(boxes[:n_boxes])
            
            features.append(padded.flatten())
            
            # Ajouter le nombre de boxes comme feature
            features.append(np.array([len(boxes)]))
        
        return np.concatenate(features)


class ShelfDetectionNet(nn.Module):
    """Réseau de neurones pour détecter la rangée modifiée"""
    
    def __init__(self, num_shelves: int = 4, max_boxes: int = 20):
        super(ShelfDetectionNet, self).__init__()
        
        # Taille d'entrée: 4 vues * (max_boxes * 4 coordonnées + 1 count)
        input_size = 4 * (max_boxes * 4 + 1)
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(512),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(128),
        )
        
        # Sortie: nombre d'objets pris par rangée (régression)
        self.regressor = nn.Linear(128, num_shelves)
        
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
              lr: float = 0.001):
        """
        Entraîne le modèle
        
        train_data: Liste de (input_dict, label_str)
        Exemple:
        [
            ({
                'left_before': [[100, 50, 20, 30], [100, 150, 20, 30]],
                'right_before': [[500, 50, 20, 30], [500, 150, 20, 30]],
                'left_after': [[100, 50, 20, 30]],
                'right_after': [[500, 50, 20, 30]]
            }, "0|-1|0|0"),
            ...
        ]
        """
        dataset = ShelfDataset(train_data, max_boxes=self.max_boxes)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            
            for features, labels in dataloader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
    
    def predict(self, input_data: dict) -> str:
        """
        Prédit la rangée modifiée et le nombre d'objets pris
        
        input_data: dict avec les clés 'left_before', 'right_before', 'left_after', 'right_after'
        Retourne: str format "0|-1|0|0"
        """
        self.model.eval()
        
        dataset = ShelfDataset([(input_data, "0|0|0|0")], max_boxes=self.max_boxes)
        features, _ = dataset[0]
        features = features.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(features)
            predictions = output.cpu().numpy()[0]
        
        # Arrondir les prédictions à l'entier le plus proche
        predictions = np.round(predictions).astype(int)
        
        return '|'.join(map(str, predictions))
    
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


# Exemple d'utilisation
if __name__ == "__main__":
    # Générer des données d'exemple
    def generate_sample_data(n_samples: int = 100) -> List[Tuple[dict, str]]:
        """Génère des données synthétiques pour tester"""
        samples = []
        
        for _ in range(n_samples):
            # Simuler 4 rangées avec des positions Y différentes
            shelf_y_positions = [50, 150, 250, 350]
            
            # Choisir une rangée aléatoire à modifier
            modified_shelf = np.random.randint(0, 4)
            n_removed = np.random.randint(1, 4)
            
            # Générer les boxes avant
            left_before = []
            right_before = []
            
            for i, y in enumerate(shelf_y_positions):
                n_items = np.random.randint(3, 8)
                for j in range(n_items):
                    x = 50 + j * 60
                    left_before.append([x, y, 20, 30])
                    right_before.append([x + 400, y, 20, 30])
            
            # Générer les boxes après (enlever des objets de la rangée modifiée)
            left_after = [b for b in left_before if not (
                shelf_y_positions[modified_shelf] - 10 < b[1] < shelf_y_positions[modified_shelf] + 10
                and np.random.random() < n_removed / 5
            )]
            
            right_after = [b for b in right_before if not (
                shelf_y_positions[modified_shelf] - 10 < b[1] < shelf_y_positions[modified_shelf] + 10
                and np.random.random() < n_removed / 5
            )]
            
            # Créer le label
            label = ['0'] * 4
            n_actual_removed = (len(left_before) - len(left_after) + 
                               len(right_before) - len(right_after)) // 2
            label[modified_shelf] = str(-n_actual_removed)
            
            samples.append(({
                'left_before': left_before,
                'right_before': right_before,
                'left_after': left_after,
                'right_after': right_after
            }, '|'.join(label)))
        
        return samples
    
    print("=== Entraînement du modèle ===")
    
    # Générer les données
    train_data = generate_sample_data(500)
    
    # Créer et entraîner le modèle
    detector = ShelfDetector(num_shelves=4, max_boxes=30)
    detector.train(train_data, epochs=50, batch_size=16, lr=0.001)
    
    # Tester sur un exemple
    print("\n=== Test de prédiction ===")
    test_input = {
        'left_before': [[50, 50, 20, 30], [110, 50, 20, 30], [50, 150, 20, 30]],
        'right_before': [[450, 50, 20, 30], [510, 50, 20, 30], [450, 150, 20, 30]],
        'left_after': [[50, 50, 20, 30], [50, 150, 20, 30]],
        'right_after': [[450, 50, 20, 30], [450, 150, 20, 30]]
    }
    
    prediction = detector.predict(test_input)
    print(f"Prédiction: {prediction}")
    
    # Sauvegarder le modèle
    detector.save('shelf_detector.pth')