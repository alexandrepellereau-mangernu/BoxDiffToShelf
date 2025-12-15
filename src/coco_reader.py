import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

class COCOReader:
    """Lecteur de fichiers d'annotations COCO JSON"""
    
    def __init__(self, json_path: str, debug: bool = False):
        """
        Initialise le lecteur avec un fichier COCO JSON
        
        Args:
            json_path: Chemin vers le fichier .coco.json
        """
        self.json_path = Path(json_path)
        with open(self.json_path, 'r') as f:
            self.data = json.load(f)
        
        # Structures pour accès rapide
        self.images_dict = {img['id']: img for img in self.data.get('images', [])}
        self.categories_dict = {cat['id']: cat for cat in self.data.get('categories', [])}
        self.annotations = self.data.get('annotations', [])
        self.debug = debug
        
        print(f"✅ Chargé: {len(self.images_dict)} images, "
              f"{len(self.annotations)} annotations, "
              f"{len(self.categories_dict)} catégories")
        
        if debug:
            print("🛠️ Mode debug activé")
            print("Aperçu des catégories:")
            for cat_id, cat_data in self.categories_dict.items():
                print(f"  - ID {cat_id}: {cat_data['name']}")
    
    def _get_camera_from_filename(self, filename: str) -> Optional[str]:
        """
        Détermine la caméra (left/right) à partir du nom de fichier
        
        Args:
            filename: Nom du fichier image
            
        Returns:
            'left', 'right' ou None si non déterminable
        """
        filename_lower = filename.lower()
        if '_right_' in filename_lower or 'right_shelf' in filename_lower:
            return 'right'
        elif '_left_' in filename_lower or 'left_shelf' in filename_lower:
            return 'left'
        return None
    
    def get_images_by_camera(self, camera: str) -> List[int]:
        """
        Récupère les IDs des images d'une caméra spécifique
        
        Args:
            camera: 'left' ou 'right'
            
        Returns:
            Liste des IDs d'images de cette caméra
        """
        assert camera in ['left', 'right'], "Camera doit être 'left' ou 'right'"

        camera = camera.lower()
        image_ids = []
        
        for img_id, img_data in self.images_dict.items():
            img_camera = self._get_camera_from_filename(img_data['file_name'])
            if img_camera == camera:
                image_ids.append(img_id)
        
        return image_ids
    
    def get_boxes_for_image(self, image_id: int = None, image_filename: str = None) -> List[Dict]:
        """
        Récupère toutes les boîtes pour une image spécifique
        
        Args:
            image_id: ID de l'image
            image_filename: Nom du fichier image (alternatif à image_id)
            
        Returns:
            Liste de dictionnaires contenant les informations des boîtes
        """
        if image_filename:
            # Extract operation ID and camera from URL
            # URL format: .../Operations/{operation_id}/PicturesAfter/raw/camera-XX.jpg
            operation_id = None
            camera_id = None
            
            if image_filename.startswith('http://') or image_filename.startswith('https://'):
                # Extract operation ID from URL path
                parts = image_filename.split('/')
                for i, part in enumerate(parts):
                    if part == 'Operations' and i + 1 < len(parts):
                        operation_id = parts[i + 1]
                    
                # Extract camera from filename
                filename = parts[-1]
                if 'camera-' in filename:
                    start_idx = filename.find('camera-')
                    end_idx = start_idx + len('camera-') + 2
                    camera_id = filename[start_idx:end_idx]
            else:
                # Direct filename
                if 'camera-' in image_filename:
                    start_idx = image_filename.find('camera-')
                    end_idx = start_idx + len('camera-') + 2
                    camera_id = image_filename[start_idx:end_idx]
            
            # Try to find matching image in COCO
            # Format in COCO: ope_{operation_id}_camera-{XX}_jpg.rf.{hash}.jpg
            for img_id, img_data in self.images_dict.items():
                file_name = img_data['file_name']
                
                # Exact filename match
                if file_name == image_filename:
                    image_id = img_id
                    break
                
                # Match by operation ID + camera
                if operation_id and camera_id:
                    if f"ope_{operation_id}_" in file_name and camera_id in file_name:
                        image_id = img_id
                        break
                
                # Fallback: match by camera only (old behavior)
                elif camera_id and camera_id in file_name:
                    image_id = img_id
                    break
            
            if image_id is None:
                raise ValueError(f"Image '{image_filename}' non trouvée (operation: {operation_id}, camera: {camera_id})")
        
        boxes = []
        for ann in self.annotations:
            if ann['image_id'] == image_id:
                # COCO format: [x, y, width, height]
                bbox = ann['bbox']
                
                box_info = {
                    'bbox': bbox,  # [x, y, width, height]
                    'category_id': ann['category_id'],
                    'category_name': self.categories_dict[ann['category_id']]['name'],
                    'area': ann.get('area', bbox[2] * bbox[3]),
                    'annotation_id': ann['id'],
                    'iscrowd': ann.get('iscrowd', 0)
                }
                boxes.append(box_info)
        
        return boxes
    
    def get_all_boxes_array(self, 
                           excluded_categories: Optional[List[str]] = None,
                           camera: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Récupère toutes les boîtes du dataset au format numpy
        
        Args:
            excluded_categories: Liste de noms de catégories à exclure
            camera: Filtre par caméra - 'left', 'right' ou None (toutes)
        
        Returns:
            Tuple de (boxes, category_ids, image_ids)
            - boxes: array (n, 4) avec [x, y, width, height]
            - category_ids: array (n,) avec les IDs de catégorie
            - image_ids: array (n,) avec les IDs d'image
        """
        # Trouver les IDs des catégories à exclure
        excluded_cat_ids = set()
        if excluded_categories:
            for cat_id, cat_data in self.categories_dict.items():
                if cat_data['name'] in excluded_categories:
                    excluded_cat_ids.add(cat_id)
        
        # Filtrer les images par caméra si nécessaire
        valid_image_ids = None
        if camera:
            valid_image_ids = set(self.get_images_by_camera(camera))
            if self.debug:
                print(f"📷 Filtrage caméra {camera}: {len(valid_image_ids)} images")
        
        boxes = []
        category_ids = []
        image_ids = []
        
        for ann in self.annotations:
            # Ignorer les annotations des catégories exclues
            if ann['category_id'] in excluded_cat_ids:
                continue
            
            # Ignorer les annotations des images non filtrées par caméra
            if valid_image_ids is not None and ann['image_id'] not in valid_image_ids:
                continue
            
            boxes.append(ann['bbox'])
            category_ids.append(ann['category_id'])
            image_ids.append(ann['image_id'])
        
        return (
            np.array(boxes),
            np.array(category_ids),
            np.array(image_ids)
        )
    
    def boxes_to_shelf_format(self, image_id: int = None, 
                             category_to_shelf: Dict[int, int] = None) -> np.ndarray:
        """
        Convertit les boîtes au format attendu par ShelfDetector
        
        Args:
            image_id: ID de l'image (None pour toutes les images)
            category_to_shelf: Dictionnaire mappant category_id -> shelf_number
                              Si None, utilise category_id comme shelf_number
        
        Returns:
            array numpy (n, 4) avec [x, y, width, height]
        """
        if image_id is not None:
            boxes = self.get_boxes_for_image(image_id)
        else:
            boxes = [{'bbox': ann['bbox'], 'category_id': ann['category_id']} 
                    for ann in self.annotations]
        
        return np.array([box['bbox'] for box in boxes])
    
    def get_labels_from_categories(self, 
                                   category_to_shelf: Dict[int, int] = None,
                                   excluded_categories: Optional[List[str]] = None,
                                   camera: Optional[str] = None) -> np.ndarray:
        """
        Récupère les labels (numéros d'étagère) à partir des catégories
        
        Args:
            category_to_shelf: Dictionnaire mappant category_id -> shelf_number
                              Si None, utilise category_id directement
            excluded_categories: Liste de noms de catégories à exclure
            camera: Filtre par caméra - 'left', 'right' ou None (toutes)
        
        Returns:
            array numpy (n,) avec les numéros d'étagère
        """
        # Trouver les IDs des catégories à exclure
        excluded_cat_ids = set()
        if excluded_categories:
            for cat_id, cat_data in self.categories_dict.items():
                if cat_data['name'] in excluded_categories:
                    excluded_cat_ids.add(cat_id)
        
        # Filtrer les images par caméra si nécessaire
        valid_image_ids = None
        if camera:
            valid_image_ids = set(self.get_images_by_camera(camera))
        
        labels = []
        for ann in self.annotations:
            cat_id = ann['category_id']
            
            # Ignorer les catégories exclues
            if cat_id in excluded_cat_ids:
                continue
            
            # Ignorer les images non filtrées par caméra
            if valid_image_ids is not None and ann['image_id'] not in valid_image_ids:
                continue
            
            if category_to_shelf:
                shelf = category_to_shelf.get(cat_id, 0)
            else:
                shelf = cat_id
            labels.append(shelf)
        
        return np.array(labels)
    
    def get_labels_from_y_position(self, 
                                   num_shelves: int = 5,
                                   excluded_categories: Optional[List[str]] = None,
                                   camera: Optional[str] = None) -> np.ndarray:
        """
        Génère les labels automatiquement basés sur la position Y des boîtes
        (utile si vos étagères ne sont pas déjà annotées par catégorie)
        
        Args:
            num_shelves: Nombre d'étagères
            excluded_categories: Liste de noms de catégories à exclure
            camera: Filtre par caméra - 'left', 'right' ou None (toutes)
        
        Returns:
            array numpy (n,) avec les numéros d'étagère estimés
        """
        boxes, _, _ = self.get_all_boxes_array(
            excluded_categories=excluded_categories,
            camera=camera
        )
        
        # Calculer le centre Y de chaque boîte
        y_centers = boxes[:, 1] + boxes[:, 3] / 2
        
        # Trouver les limites Y
        y_min, y_max = y_centers.min(), y_centers.max()
        
        # Diviser en étagères
        labels = np.floor((y_centers - y_min) / (y_max - y_min) * num_shelves)
        labels = np.clip(labels, 0, num_shelves - 1).astype(int)
        
        return labels
    
    def get_all_labels_name(self) -> Dict[int, str]:
        """
        Récupère les noms des labels (catégories)
        """
        return {k: v['name'] for k, v in self.categories_dict.items()}
    
    def filter_by_category(self, category_names: List[str]) -> List[Dict]:
        """
        Filtre les annotations par nom de catégorie
        
        Args:
            category_names: Liste de noms de catégories à garder
        
        Returns:
            Liste des annotations filtrées
        """
        cat_ids = [cat_id for cat_id, cat_data in self.categories_dict.items() 
                   if cat_data['name'] in category_names]
        
        filtered = [ann for ann in self.annotations if ann['category_id'] in cat_ids]
        return filtered
    
    def get_image_info(self, image_id: int) -> Dict:
        """Récupère les informations d'une image"""
        return self.images_dict.get(image_id)
    
    def get_statistics(self, camera: Optional[str] = None) -> Dict:
        """
        Récupère des statistiques sur le dataset
        
        Args:
            camera: Filtre par caméra - 'left', 'right' ou None (toutes)
        """
        boxes, cat_ids, img_ids = self.get_all_boxes_array(camera=camera)
        
        # Compter les images selon le filtre caméra
        if camera:
            total_images = len(self.get_images_by_camera(camera))
        else:
            total_images = len(self.images_dict)
        
        stats = {
            'total_images': total_images,
            'total_annotations': len(boxes),
            'total_categories': len(self.categories_dict),
            'annotations_per_image': len(boxes) / max(total_images, 1),
            'categories': {cat_data['name']: np.sum(cat_ids == cat_id) 
                          for cat_id, cat_data in self.categories_dict.items()},
            'bbox_stats': {
                'width_mean': boxes[:, 2].mean() if len(boxes) > 0 else 0,
                'height_mean': boxes[:, 3].mean() if len(boxes) > 0 else 0,
                'width_std': boxes[:, 2].std() if len(boxes) > 0 else 0,
                'height_std': boxes[:, 3].std() if len(boxes) > 0 else 0,
            }
        }
        
        if camera:
            stats['camera'] = camera
        
        return stats
    
    def export_for_training(self, 
                           output_path: str = None, 
                           category_to_shelf: Dict[int, int] = None,
                           use_y_position: bool = False,
                           num_shelves: int = 5,
                           excluded_categories: Optional[List[str]] = None,
                           camera: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Exporte les données au format prêt pour l'entraînement
        
        Args:
            output_path: Chemin de sauvegarde (optionnel)
            category_to_shelf: Mapping catégorie -> étagère
            use_y_position: Si True, génère les labels depuis la position Y
            num_shelves: Nombre d'étagères (si use_y_position=True)
            excluded_categories: Liste de noms de catégories à exclure (ex: ['black-list'])
            camera: Filtre par caméra - 'left', 'right' ou None (toutes)
        
        Returns:
            Tuple (boxes, labels)
        """
        boxes, _, _ = self.get_all_boxes_array(
            excluded_categories=excluded_categories,
            camera=camera
        )
        
        if use_y_position:
            labels = self.get_labels_from_y_position(
                num_shelves, 
                excluded_categories=excluded_categories,
                camera=camera
            )
        else:
            labels = self.get_labels_from_categories(
                category_to_shelf, 
                excluded_categories=excluded_categories,
                camera=camera
            )
        
        if camera:
            print(f"📷 Caméra: {camera}")
        print(f"   {len(self.get_images_by_camera(camera)) if camera else len(self.images_dict)} images restantes après filtrage")
        if excluded_categories:
            print(f"⚠️  Catégories exclues: {', '.join(excluded_categories)}")
        print(f"   {len(boxes)} annotations restantes après filtrage")
        
        if output_path:
            np.savez(output_path, boxes=boxes, labels=labels)
            print(f"✓ Données exportées vers {output_path}")
        
        return boxes, labels
    
    def prepare_image_pairs(self, csv_file: str, camera=None) -> Tuple[List[Tuple[str, str]], List[str]]:
        """
        Prépare des paires d'images avant/après à partir d'un fichier CSV
        
        Args:
            csv_file: Chemin vers le fichier CSV avec colonnes 'before', 'after'
        
        Returns:
            Liste de tuples (before_image_name, after_image_name, result)
        """
        import pandas as pd
        
        df = pd.read_csv(csv_file)
        image_pairs = []
        results = []
        
        for _, row in df.iterrows():
            left_before_name = row['PictureLeftBefore']
            left_after_name = row['PictureLeftAfter']
            right_after_name = row['PictureRightAfter']
            right_before_name = row['PictureRightBefore']
            
            # When camera is None, process both left and right
            # When camera is specified, only process that camera
            if camera is None or camera == 'right':
                image_pairs.append((right_before_name, right_after_name))
                results.append(row['ShelfReview'])
            if camera is None or camera == 'left':
                image_pairs.append((left_before_name, left_after_name))
                results.append(row['ShelfReview'])
        
        return image_pairs, results 

    def images_to_boxes(self, images_names: List[Tuple[str, str]]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Exporte les données au format avant/après pour ShelfDetector
        
        Args:
            images_names: Liste de tuples (before_image_name, after_image_name)
        
        Returns:
            boxes
        """
        boxes = []
        skipped = 0
        
        if self.debug:
            print(f"🔍 Processing {len(images_names)} image pairs...")
        
        for before_name, after_name in images_names:
            try:
                before_boxes = self.get_boxes_for_image(image_filename=before_name)
                after_boxes = self.get_boxes_for_image(image_filename=after_name)

                # Ignorer si aucune boîte dans avant ou après
                if len(before_boxes) == 0 or len(after_boxes) == 0:
                    skipped += 1
                    continue

                boxes.append((np.array([box['bbox'] for box in before_boxes]),
                              np.array([box['bbox'] for box in after_boxes])))
            except (ValueError, KeyError) as e:
                # Image not found in COCO annotations
                skipped += 1
                if self.debug:
                    print(f"  ⚠️  Skipped pair ({before_name}, {after_name}): {e}")
                continue
        
        if self.debug:
            print(f"  ✅ Kept {len(boxes)} pairs, skipped {skipped} pairs")
        
        return boxes

    def export_for_training_before_after(self, csv_file: str, camera=None) -> Tuple[np.ndarray, List[str]]:
        """
        Exporte les données au format avant/après pour ShelfDetector
        
        Args:
            csv_file: Chemin vers le fichier CSV avec colonnes 'before', 'after'
            camera: Filtre par caméra - 'left', 'right' ou None (toutes)
        
        Returns:
            boxes
        """
        image_pairs, labels = self.prepare_image_pairs(csv_file, camera=camera)
        boxes = self.images_to_boxes(image_pairs)
        return boxes, labels
