import numpy as np
from typing import List, Tuple

from shelf_detector import ShelfDetector

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
        
        # Générer boxes AVANT (combinant left et right)
        before = []
        
        for shelf_idx, (y_pos, n_items) in enumerate(zip(shelf_positions, items_per_shelf)):
            for j in range(n_items):
                x = 80 + j * 70 + np.random.randint(-5, 5)
                box_h = 40 + np.random.randint(-5, 5)
                box_w = 50 + np.random.randint(-5, 5)
                
                before.append([x, y_pos, box_h, box_w])
        
        # Générer boxes APRÈS (retirer exactement n_removed de la rangée modifiée)
        after = []
        removed_count = 0
        
        for box in before:
            y = box[1]
            shelf_idx = min(range(4), key=lambda i: abs(y - shelf_positions[i]))
            
            if shelf_idx == modified_shelf and removed_count < n_removed:
                removed_count += 1
                continue
            after.append(box)
        
        # Label
        label = ['0'] * 4
        label[modified_shelf] = str(-n_removed)
        
        sample = ({
            'before': before,
            'after': after
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
    # Test with boxes distributed across all shelves (like in training)
    test_input = {
        'before': [
            # Shelf 0 (Y=50) - 3 items
            [80, 50, 40, 50], [150, 50, 40, 50], [220, 50, 40, 50],
            # Shelf 1 (Y=150) - 4 items
            [80, 150, 40, 50], [150, 150, 40, 50], [220, 150, 40, 50], [290, 150, 40, 50],
            # Shelf 2 (Y=250) - 5 items
            [80, 250, 40, 50], [150, 250, 40, 50], [220, 250, 40, 50], [290, 250, 40, 50], [360, 250, 40, 50],
            # Shelf 3 (Y=350) - 4 items
            [80, 350, 40, 50], [150, 350, 40, 50], [220, 350, 40, 50], [290, 350, 40, 50]
        ],
        'after': [
            # Shelf 0 (Y=50) - 2 items (removed 1)
            [80, 50, 40, 50], [220, 50, 40, 50],
            # Shelf 1 (Y=150) - 4 items (no change)
            [80, 150, 40, 50], [150, 150, 40, 50], [220, 150, 40, 50], [290, 150, 40, 50],
            # Shelf 2 (Y=250) - 5 items (no change)
            [80, 250, 40, 50], [150, 250, 40, 50], [220, 250, 40, 50], [290, 250, 40, 50], [360, 250, 40, 50],
            # Shelf 3 (Y=350) - 4 items (no change)
            [80, 350, 40, 50], [150, 350, 40, 50], [220, 350, 40, 50], [290, 350, 40, 50]
        ]
    }
    
    prediction = detector.predict(test_input)
    print(f"Prédiction: {prediction}")
    print("Attendu: -1|0|0|0")
    
    detector.save('models/shelf_detector.pth')