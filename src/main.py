#!/usr/bin/env python3
"""
Main training script using COCO annotations
Reads boxes from data/train/_annotations.coco.json and trains a shelf detection model
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import List, Tuple

from coco_reader import COCOReader
from shelf_detector import ShelfDetector


def prepare_training_data_from_coco(coco_path: str,
                                     csv_path: str,
                                     num_shelves: int = 4,
                                     camera: str = None,
                                     excluded_categories: List[str] = None) -> List[Tuple[dict, str]]:
    """
    Prepare training data from COCO annotations using real before/after image pairs
    
    Args:
        coco_path: Path to COCO JSON file
        csv_path: Path to CSV file with before/after image pairs
        num_shelves: Number of shelves to detect
        camera: Filter by camera ('left', 'right', or None for all)
        excluded_categories: Categories to exclude from training
    
    Returns:
        List of training samples in the format expected by ShelfDetector
    """
    print(f"📁 Loading COCO annotations from: {coco_path}")
    coco = COCOReader(coco_path, debug=True)
    
    # Get statistics
    stats = coco.get_statistics(camera=camera)
    print(f"\n📊 Dataset Statistics:")
    print(f"  Total images: {stats['total_images']}")
    print(f"  Total annotations: {stats['total_annotations']}")
    print(f"  Annotations per image: {stats['annotations_per_image']:.2f}")
    print(f"  Categories: {list(stats['categories'].keys())}")
    
    # Prepare image pairs from CSV
    print(f"\n📋 Loading image pairs from CSV: {csv_path}")
    boxes_pairs, labels = coco.export_for_training_before_after(csv_path, camera=camera)
    print(f"\n✅ Loaded {len(boxes_pairs)} boxes pairs with {len(labels)} labels")
    print(f"  Sample boxes pairs: {boxes_pairs[:3]}")
    print(f"  Unique labels: {np.unique(labels)}")
    
    print(f"  Unique shelf labels: {np.unique(labels)}")
    
    # Convert to ShelfDetector format
    training_samples = []
    
    print(f"\n🔧 Creating training samples from real image pairs...")
    
    for i in range(len(boxes_pairs)):
        before_boxes_array = boxes_pairs[i][0]
        after_boxes_array = boxes_pairs[i][1]
        label = labels[i]
        print(f"  before boxes: {before_boxes_array}, after boxes: {after_boxes_array}, label: {label}")

        # Create sample
        sample = (
            {'before': before_boxes_array.tolist(), 'after': after_boxes_array.tolist()},
            '|'.join(label)
        )
        training_samples.append(sample)
    
    print(f"  ✅ Generated {len(training_samples)} training samples from real image pairs")
    
    return training_samples


def split_train_val(samples: List, val_split: float = 0.2) -> Tuple[List, List]:
    """Split data into training and validation sets"""
    n_val = int(len(samples) * val_split)
    n_train = len(samples) - n_val
    
    # Shuffle samples
    indices = np.random.permutation(len(samples))
    train_samples = [samples[i] for i in indices[:n_train]]
    val_samples = [samples[i] for i in indices[n_train:]]
    
    return train_samples, val_samples


def main():
    """Main training pipeline"""
    # Configuration
    COCO_PATH = "../data/train/_annotations.coco.json"
    CSV_PATH = "../data/dataset.csv"
    MODEL_SAVE_PATH = "../models/shelf_detector.pth"
    NUM_SHELVES = 4
    MAX_BOXES = 20
    EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    CAMERA = None  # Use None for all cameras, or 'left'/'right' for specific camera
    EXCLUDED_CATEGORIES = None  # e.g., ['black-list', 'test']
    
    print("=" * 60)
    print("🚀 Shelf Detection Training Pipeline")
    print("=" * 60)
    
    # Get absolute paths
    script_dir = Path(__file__).parent
    coco_path = (script_dir / COCO_PATH).resolve()
    csv_path = (script_dir / CSV_PATH).resolve()
    
    if not coco_path.exists():
        print(f"❌ Error: COCO file not found at {coco_path}")
        sys.exit(1)
    
    if not csv_path.exists():
        print(f"❌ Error: CSV file not found at {csv_path}")
        sys.exit(1)
    
    # Prepare training data
    print("\n" + "=" * 60)
    print("📦 Preparing Training Data")
    print("=" * 60)
    
    all_samples = prepare_training_data_from_coco(
        str(coco_path),
        str(csv_path),
        num_shelves=NUM_SHELVES,
        camera=CAMERA,
        excluded_categories=EXCLUDED_CATEGORIES
    )
    
    if len(all_samples) == 0:
        print("❌ Error: No training samples generated!")
        print("   Check your COCO annotations file and parameters.")
        sys.exit(1)
    
    # Split into train and validation
    train_samples, val_samples = split_train_val(all_samples, val_split=0.2)
    
    print(f"\n📊 Data Split:")
    print(f"  Training samples: {len(train_samples)}")
    print(f"  Validation samples: {len(val_samples)}")
    
    if len(val_samples) == 0:
        print("  ⚠️  Warning: No validation samples. Using training samples for validation.")
        val_samples = train_samples[:min(10, len(train_samples))]
    
    # Initialize model
    print("\n" + "=" * 60)
    print("🧠 Initializing Model")
    print("=" * 60)
    
    detector = ShelfDetector(num_shelves=NUM_SHELVES, max_boxes=MAX_BOXES)
    print(f"  Model: ShelfDetectionNet")
    print(f"  Device: {detector.device}")
    print(f"  Shelves: {NUM_SHELVES}")
    print(f"  Max boxes: {MAX_BOXES}")
    
    # Train model
    print("\n" + "=" * 60)
    print("🎯 Training Model")
    print("=" * 60)
    
    detector.train(
        train_data=train_samples,
        validation_data=val_samples,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE
    )
    
    # Evaluate model
    print("\n" + "=" * 60)
    print("📈 Evaluation")
    print("=" * 60)
    
    metrics = detector.evaluate(val_samples)
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  MSE: {metrics['mse']:.4f}")
    print(f"  Shelf Accuracy: {metrics['shelf_accuracy']*100:.2f}%")
    print(f"  Exact Accuracy: {metrics['exact_accuracy']*100:.2f}%")
    
    # Save model
    print("\n" + "=" * 60)
    print("💾 Saving Model")
    print("=" * 60)
    
    model_path = (script_dir / MODEL_SAVE_PATH).resolve()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    detector.save(str(model_path))
    print(f"  Model saved to: {model_path}")
    
    # Test prediction
    if len(val_samples) > 0:
        print("\n" + "=" * 60)
        print("🔮 Test Prediction")
        print("=" * 60)
        
        test_sample = val_samples[0]
        test_input, true_label = test_sample
        
        prediction = detector.predict(test_input)
        print(f"  True label: {true_label}")
        print(f"  Prediction: {prediction}")
    
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    import torch
    torch.manual_seed(42)
    
    main()
