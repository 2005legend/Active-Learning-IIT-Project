import torch
import torch.nn as nn
from torchvision import models

class WildlifeResNet(nn.Module):
    """ResNet18 for multi-class wildlife classification"""
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        # Load pretrained ResNet18
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        
        # Replace final layer for wildlife classes
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)
        
        self.num_classes = num_classes
    
    def forward(self, x):
        return self.backbone(x)
    
    def extract_features(self, x):
        """Extract features from penultimate layer"""
        # Get all layers except the final FC layer
        modules = list(self.backbone.children())[:-1]
        feature_extractor = nn.Sequential(*modules)
        
        with torch.no_grad():
            features = feature_extractor(x)
            features = features.view(features.size(0), -1)  # Flatten
        
        return features

class WildlifeClassifier:
    """Wrapper class for wildlife classification"""
    
    def __init__(self, class_names, device='cuda'):
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.device = device
        self.model = None
    
    def create_model(self, pretrained=True):
        """Create the model"""
        self.model = WildlifeResNet(self.num_classes, pretrained=pretrained)
        self.model = self.model.to(self.device)
        return self.model
    
    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint"""
        if self.model is None:
            self.create_model()
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        return self.model
    
    def predict(self, image_tensor):
        """Make prediction on image tensor"""
        if self.model is None:
            raise ValueError("Model not loaded. Call create_model() or load_checkpoint() first.")
        
        self.model.eval()
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get top prediction
            confidence, predicted = torch.max(probabilities, 1)
            
            predicted_class = self.class_names[predicted.item()]
            confidence_score = confidence.item()
            
            return predicted_class, confidence_score, probabilities[0].cpu().numpy()
    
    def get_top_k_predictions(self, image_tensor, k=3):
        """Get top-k predictions"""
        if self.model is None:
            raise ValueError("Model not loaded.")
        
        self.model.eval()
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get top-k predictions
            top_probs, top_indices = torch.topk(probabilities, k, dim=1)
            
            results = []
            for i in range(k):
                class_name = self.class_names[top_indices[0][i].item()]
                confidence = top_probs[0][i].item()
                results.append((class_name, confidence))
            
            return results

# Quick model factory function
def create_wildlife_model(class_names, pretrained=True, device='cuda'):
    """Factory function to create wildlife model"""
    classifier = WildlifeClassifier(class_names, device)
    model = classifier.create_model(pretrained)
    return model, classifier