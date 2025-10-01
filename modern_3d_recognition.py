"""
Modern 3D Object Recognition System
====================================

This project implements a state-of-the-art 3D object recognition system using:
- Voxel-based 3D CNN with modern architectures
- Synthetic dataset generation for training
- Web-based UI for interactive classification
- Comprehensive visualization tools
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# 3D processing libraries
try:
    import trimesh
    import open3d as o3d
    import pyvista as pv
    HAS_3D_LIBS = True
except ImportError:
    HAS_3D_LIBS = False
    print("Warning: 3D libraries not installed. Install with: pip install trimesh open3d pyvista")

# Visualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Configuration class for the 3D object recognition system"""
    # Model parameters
    voxel_size: int = 64
    num_classes: int = 10
    batch_size: int = 16
    learning_rate: float = 0.001
    num_epochs: int = 50
    
    # Data parameters
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Model architecture
    conv_channels: List[int] = None
    fc_layers: List[int] = None
    dropout_rate: float = 0.3
    
    # Training
    device: str = 'auto'
    save_best_model: bool = True
    early_stopping_patience: int = 10
    
    def __post_init__(self):
        if self.conv_channels is None:
            self.conv_channels = [32, 64, 128, 256]
        if self.fc_layers is None:
            self.fc_layers = [512, 256]
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Synthetic3DDataset(Dataset):
    """Synthetic 3D dataset generator for training"""
    
    def __init__(self, num_samples: int = 1000, voxel_size: int = 64, num_classes: int = 10):
        self.num_samples = num_samples
        self.voxel_size = voxel_size
        self.num_classes = num_classes
        self.class_names = [
            'chair', 'table', 'sofa', 'bed', 'desk',
            'bookshelf', 'dresser', 'nightstand', 'wardrobe', 'cabinet'
        ]
        
    def generate_primitive_shape(self, shape_type: str, size: Tuple[int, int, int]) -> np.ndarray:
        """Generate basic 3D primitive shapes"""
        voxel = np.zeros((self.voxel_size, self.voxel_size, self.voxel_size), dtype=np.float32)
        
        # Center coordinates
        cx, cy, cz = self.voxel_size // 2, self.voxel_size // 2, self.voxel_size // 2
        
        if shape_type == 'box':
            # Rectangular box
            w, h, d = size
            voxel[cx-w//2:cx+w//2, cy-h//2:cy+h//2, cz-d//2:cz+d//2] = 1.0
            
        elif shape_type == 'cylinder':
            # Cylindrical shape
            r, h = size[0], size[2]
            for x in range(self.voxel_size):
                for y in range(self.voxel_size):
                    for z in range(self.voxel_size):
                        if (x - cx)**2 + (y - cy)**2 <= r**2 and abs(z - cz) <= h//2:
                            voxel[x, y, z] = 1.0
                            
        elif shape_type == 'sphere':
            # Spherical shape
            r = size[0]
            for x in range(self.voxel_size):
                for y in range(self.voxel_size):
                    for z in range(self.voxel_size):
                        if (x - cx)**2 + (y - cy)**2 + (z - cz)**2 <= r**2:
                            voxel[x, y, z] = 1.0
                            
        elif shape_type == 'chair':
            # Chair-like structure
            # Seat
            voxel[cx-8:cx+8, cy-2:cy+2, cz-8:cz+8] = 1.0
            # Back
            voxel[cx-8:cx+8, cy+2:cy+12, cz+6:cz+8] = 1.0
            # Legs
            voxel[cx-6:cx-4, cy-12:cy-2, cz-6:cz-4] = 1.0
            voxel[cx+4:cx+6, cy-12:cy-2, cz-6:cz-4] = 1.0
            voxel[cx-6:cx-4, cy-12:cy-2, cz+4:cz+6] = 1.0
            voxel[cx+4:cx+6, cy-12:cy-2, cz+4:cz+6] = 1.0
            
        elif shape_type == 'table':
            # Table-like structure
            # Top
            voxel[cx-12:cx+12, cy+8:cy+10, cz-12:cz+12] = 1.0
            # Legs
            voxel[cx-10:cx-8, cy-8:cy+8, cz-10:cz-8] = 1.0
            voxel[cx+8:cx+10, cy-8:cy+8, cz-10:cz-8] = 1.0
            voxel[cx-10:cx-8, cy-8:cy+8, cz+8:cz+10] = 1.0
            voxel[cx+8:cx+10, cy-8:cy+8, cz+8:cz+10] = 1.0
            
        return voxel
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random class and shape parameters
        class_id = idx % self.num_classes
        shape_type = self.class_names[class_id]
        
        # Add noise and variations
        noise_factor = np.random.uniform(0.8, 1.2)
        size_variation = np.random.uniform(0.7, 1.3)
        
        # Generate base size based on class
        base_sizes = {
            'chair': (16, 20, 16),
            'table': (24, 20, 24),
            'sofa': (32, 12, 16),
            'bed': (40, 8, 20),
            'desk': (20, 16, 12),
            'bookshelf': (12, 32, 16),
            'dresser': (20, 16, 12),
            'nightstand': (12, 12, 12),
            'wardrobe': (16, 32, 12),
            'cabinet': (16, 20, 12)
        }
        
        base_size = base_sizes.get(shape_type, (16, 16, 16))
        size = tuple(int(s * size_variation) for s in base_size)
        
        # Generate voxel grid
        voxel = self.generate_primitive_shape(shape_type, size)
        
        # Add noise
        noise = np.random.normal(0, 0.1, voxel.shape)
        voxel = np.clip(voxel + noise, 0, 1)
        
        return torch.tensor(voxel, dtype=torch.float32), torch.tensor(class_id, dtype=torch.long)

class Modern3DCNN(nn.Module):
    """Modern 3D CNN architecture with advanced techniques"""
    
    def __init__(self, config: Config):
        super(Modern3DCNN, self).__init__()
        self.config = config
        
        # Convolutional layers with batch normalization and dropout
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        
        in_channels = 1
        for i, out_channels in enumerate(config.conv_channels):
            self.conv_layers.append(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1))
            self.bn_layers.append(nn.BatchNorm3d(out_channels))
            in_channels = out_channels
        
        # Calculate flattened size after convolutions
        self.pool = nn.MaxPool3d(2)
        self.dropout = nn.Dropout3d(config.dropout_rate)
        
        # Calculate the size after convolutions and pooling
        conv_output_size = self._calculate_conv_output_size()
        
        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        fc_input_size = conv_output_size
        
        for fc_size in config.fc_layers:
            self.fc_layers.append(nn.Linear(fc_input_size, fc_size))
            fc_input_size = fc_size
        
        self.classifier = nn.Linear(fc_input_size, config.num_classes)
        
    def _calculate_conv_output_size(self):
        """Calculate the output size after convolutional layers"""
        size = self.config.voxel_size
        for _ in range(len(self.conv_layers)):
            size = size // 2  # MaxPool3d reduces size by 2
        return self.config.conv_channels[-1] * size * size * size
    
    def forward(self, x):
        # Convolutional layers
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = F.relu(bn(conv(x)))
            x = self.pool(x)
            x = self.dropout(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        for fc in self.fc_layers:
            x = F.relu(fc(x))
            x = F.dropout(x, p=self.config.dropout_rate, training=self.training)
        
        # Classification
        x = self.classifier(x)
        return x

class ModelTrainer:
    """Training and evaluation class"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)
        self.model = Modern3DCNN(config).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=5, factor=0.5)
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training")):
            data, target = data.to(self.device), target.to(self.device)
            
            # Add channel dimension
            data = data.unsqueeze(1)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validating"):
                data, target = data.to(self.device), target.to(self.device)
                data = data.unsqueeze(1)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """Complete training loop"""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if self.config.save_best_model:
                    torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }

class VisualizationTools:
    """Tools for visualizing 3D objects and results"""
    
    @staticmethod
    def plot_training_history(history: Dict):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(history['train_losses'], label='Train Loss')
        ax1.plot(history['val_losses'], label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(history['train_accuracies'], label='Train Accuracy')
        ax2.plot(history['val_accuracies'], label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def visualize_voxel_grid(voxel: np.ndarray, title: str = "3D Voxel Grid"):
        """Visualize 3D voxel grid using plotly"""
        if not HAS_3D_LIBS:
            print("3D visualization libraries not available")
            return
        
        # Create voxel coordinates
        x, y, z = np.where(voxel > 0.5)
        
        fig = go.Figure(data=go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=2,
                color=voxel[x, y, z],
                colorscale='Viridis',
                opacity=0.8
            )
        ))
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )
        
        fig.show()
    
    @staticmethod
    def plot_confusion_matrix(y_true: List, y_pred: List, class_names: List[str]):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main function to run the 3D object recognition system"""
    # Load configuration
    config = Config()
    
    logger.info("Initializing 3D Object Recognition System")
    logger.info(f"Device: {config.device}")
    logger.info(f"Voxel size: {config.voxel_size}")
    logger.info(f"Number of classes: {config.num_classes}")
    
    # Create dataset
    logger.info("Creating synthetic dataset...")
    dataset = Synthetic3DDataset(
        num_samples=2000,
        voxel_size=config.voxel_size,
        num_classes=config.num_classes
    )
    
    # Split dataset
    train_size = int(config.train_split * len(dataset))
    val_size = int(config.val_split * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    logger.info(f"Dataset split - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Initialize trainer
    trainer = ModelTrainer(config)
    
    # Train model
    logger.info("Starting training...")
    history = trainer.train(train_loader, val_loader)
    
    # Plot training history
    VisualizationTools.plot_training_history(history)
    
    # Test model
    logger.info("Testing model...")
    test_loss, test_acc = trainer.validate(test_loader)
    logger.info(f"Test Accuracy: {test_acc:.2f}%")
    
    # Generate predictions for confusion matrix
    trainer.model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(config.device), target.to(config.device)
            data = data.unsqueeze(1)
            output = trainer.model(data)
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Plot confusion matrix
    VisualizationTools.plot_confusion_matrix(
        all_targets, all_preds, dataset.class_names
    )
    
    # Save model and results
    torch.save(trainer.model.state_dict(), 'final_model.pth')
    
    # Save configuration
    with open('config.yaml', 'w') as f:
        yaml.dump(config.__dict__, f, default_flow_style=False)
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
