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
                                     num_shelves: int = 4,
                                     camera: str = None,
                                     excluded_categories: List[str] = None,
                                     samples_per_shelf: int = 50) -> List[Tuple[dict, str]]:
    """
    Prepare training data from COCO annotations
    
    Args:
        coco_path: Path to COCO JSON file
        num_shelves: Number of shelves to detect
        camera: Filter by camera ('left', 'right', or None for all)
        excluded_categories: Categories to exclude from training
        samples_per_shelf: Number of samples to generate per shelf
    
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
    
    # Export boxes and labels
    print(f"\n🔄 Exporting training data...")
    boxes, labels = coco.export_for_training(
        camera=camera,
        use_y_position=True,  # Generate labels based on Y position
        num_shelves=num_shelves,
        excluded_categories=excluded_categories
    )
    
    print(f"\n✅ Loaded {len(boxes)} boxes with {len(labels)} labels")
    print(f"  Unique shelf labels: {np.unique(labels)}")
    
    # Convert to ShelfDetector format by generating synthetic before/after pairs
    training_samples = []
    
    # Group boxes by shelf
    boxes_by_shelf = {}
    for shelf_id in range(num_shelves):
        boxes_by_shelf[shelf_id] = boxes[labels == shelf_id].tolist()
    
    print(f"\n🔧 Generating synthetic training samples...")
    print(f"  Samples per shelf: {samples_per_shelf}")
    
    # Generate samples for each shelf
    for target_shelf in range(num_shelves):
        shelf_boxes = boxes_by_shelf[target_shelf]
        if len(shelf_boxes) < 2:
            print(f"  ⚠️  Shelf {target_shelf}: Not enough boxes ({len(shelf_boxes)}), skipping")
            continue
        
        for _ in range(samples_per_shelf):
            # Create "before" state with boxes from all shelves
            before_boxes = []
            for shelf_id in range(num_shelves):
                if len(boxes_by_shelf[shelf_id]) > 0:
                    # Randomly sample some boxes from this shelf
                    n_boxes = np.random.randint(1, min(8, len(boxes_by_shelf[shelf_id]) + 1))
                    indices = np.random.choice(len(boxes_by_shelf[shelf_id]), n_boxes, replace=False)
                    before_boxes.extend([boxes_by_shelf[shelf_id][i] for i in indices])
            
            # Create "after" state by removing boxes from target shelf
            n_remove = np.random.randint(1, min(4, len(shelf_boxes) // 2 + 1))
            
            # Count boxes from target shelf in before state
            target_shelf_count = sum(1 for box in before_boxes 
                                    if any(np.allclose(box, sb) for sb in shelf_boxes))
            
            # Make sure we don't remove more than what's in the before state
            n_remove = min(n_remove, target_shelf_count)
            
            if n_remove == 0:
                continue
            
            # Remove n_remove boxes from target shelf
            after_boxes = []
            removed = 0
            for box in before_boxes:
                is_target_shelf = any(np.allclose(box, sb) for sb in shelf_boxes)
                if is_target_shelf and removed < n_remove:
                    removed += 1
                    continue
                after_boxes.append(box)
            
            # Create label
            label = ['0'] * num_shelves
            label[target_shelf] = str(-n_remove)
            
            sample = (
                {'before': before_boxes, 'after': after_boxes},
                '|'.join(label)
            )
            training_samples.append(sample)
    
    print(f"  ✅ Generated {len(training_samples)} training samples")
    
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
    
    # Get absolute path to COCO file
    script_dir = Path(__file__).parent
    coco_path = (script_dir / COCO_PATH).resolve()
    
    if not coco_path.exists():
        print(f"❌ Error: COCO file not found at {coco_path}")
        sys.exit(1)
    
    # Prepare training data
    print("\n" + "=" * 60)
    print("📦 Preparing Training Data")
    print("=" * 60)
    
    all_samples = prepare_training_data_from_coco(
        str(coco_path),
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
