import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from typing import List, Tuple, Optional

class WildlifeDataset:
    """CIFAR-100 Wildlife subset for animal classification"""
    
    # CIFAR-100 animal classes mapping
    ANIMAL_CLASSES = {
        'aquatic_mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
        'fish': ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
        'large_carnivores': ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
        'large_herbivores': ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
        'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
        'small_mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel']
    }
    
    # Flatten to get all animal class names
    ALL_ANIMALS = []
    for category in ANIMAL_CLASSES.values():
        ALL_ANIMALS.extend(category)
    
    def __init__(self, root: str = './data', train: bool = True, 
                 transform: Optional[transforms.Compose] = None,
                 animal_subset: Optional[List[str]] = None):
        """
        Initialize Wildlife Dataset from CIFAR-100
        
        Args:
            root: Data directory
            train: Training or test split
            transform: Image transformations
            animal_subset: Specific animals to include (None = all animals)
        """
        self.root = root
        self.train = train
        self.animal_subset = animal_subset or self.ALL_ANIMALS
        
        # Load CIFAR-100
        self.cifar100 = torchvision.datasets.CIFAR100(
            root=root, train=train, download=True, transform=transform
        )
        
        # Get CIFAR-100 class names
        self.cifar100_classes = [
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
            'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
            'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
            'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
            'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
            'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
            'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
            'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
            'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
            'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
            'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
            'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
            'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
            'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
            'worm'
        ]
        
        # Create mapping from CIFAR-100 indices to animal indices
        self.animal_indices = []
        self.animal_class_names = []
        
        for i, class_name in enumerate(self.cifar100_classes):
            if class_name in self.animal_subset:
                self.animal_indices.append(i)
                self.animal_class_names.append(class_name)
        
        # Filter dataset to only include animal classes
        self.filtered_indices = []
        for idx, (_, label) in enumerate(self.cifar100):
            if label in self.animal_indices:
                self.filtered_indices.append(idx)
        
        # Create label mapping (CIFAR-100 label -> Wildlife label)
        self.label_mapping = {}
        for new_idx, old_idx in enumerate(self.animal_indices):
            self.label_mapping[old_idx] = new_idx
        
        print(f"Wildlife Dataset: {len(self.animal_class_names)} classes, {len(self.filtered_indices)} samples")
        print(f"Classes: {self.animal_class_names}")
    
    def __len__(self):
        return len(self.filtered_indices)
    
    def __getitem__(self, idx):
        # Get original CIFAR-100 sample
        cifar_idx = self.filtered_indices[idx]
        image, original_label = self.cifar100[cifar_idx]
        
        # Map to wildlife label
        wildlife_label = self.label_mapping[original_label]
        
        return image, wildlife_label
    
    def get_class_names(self):
        return self.animal_class_names
    
    def get_num_classes(self):
        return len(self.animal_class_names)

def get_wildlife_transforms(image_size: int = 64):
    """Get transforms for wildlife dataset"""
    
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, test_transform

def create_wildlife_dataloaders(root: str = './data', 
                              batch_size: int = 32,
                              image_size: int = 64,
                              animal_subset: Optional[List[str]] = None):
    """Create wildlife dataloaders"""
    
    train_transform, test_transform = get_wildlife_transforms(image_size)
    
    train_dataset = WildlifeDataset(
        root=root, train=True, transform=train_transform, animal_subset=animal_subset
    )
    
    test_dataset = WildlifeDataset(
        root=root, train=False, transform=test_transform, animal_subset=animal_subset
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    return train_loader, test_loader, train_dataset.get_class_names()

if __name__ == "__main__":
    # Test the dataset
    print("Testing Wildlife Dataset...")
    
    # Test with subset of animals
    subset = ['bear', 'tiger', 'wolf', 'elephant', 'dolphin', 'snake']
    train_loader, test_loader, class_names = create_wildlife_dataloaders(
        animal_subset=subset, batch_size=16
    )
    
    print(f"Classes: {class_names}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test a batch
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}")
        print(f"Labels: {labels}")
        break