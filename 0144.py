"""
Project 144: Modern 3D Object Recognition System
=================================================

This is the original implementation, now updated with modern techniques.
For the full-featured system, see modern_3d_recognition.py

Description:
3D object recognition involves identifying and classifying objects based on their 
three-dimensional representations. This project explores using voxel-based 
representation (3D grids) and a 3D Convolutional Neural Network (3D CNN) to 
classify 3D objects from volumetric data.

Updated Implementation: Modern Voxel-Based 3D Object Recognition with PyTorch
Install required libraries: pip install -r requirements.txt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# Try to import 3D processing libraries
try:
    import trimesh
    import matplotlib.pyplot as plt
    from skimage import measure
    from scipy.ndimage import zoom
    HAS_3D_LIBS = True
except ImportError:
    HAS_3D_LIBS = False
    print("Warning: Some 3D libraries not available. Install with: pip install trimesh scikit-image scipy")

def mesh_to_voxel(file_path, grid_size=32):
    """Convert mesh to voxel grid with error handling"""
    if not HAS_3D_LIBS:
        print("3D libraries not available. Creating dummy voxel...")
        return np.random.rand(grid_size, grid_size, grid_size).astype(np.float32)
    
    try:
        mesh = trimesh.load(file_path)
        voxel = mesh.voxelized(pitch=mesh.scale / grid_size).matrix.astype(np.float32)
        voxel = zoom(voxel, (grid_size / voxel.shape[0],) * 3)  # Normalize to fixed size
        return voxel
    except Exception as e:
        print(f"Error loading mesh: {e}. Creating dummy voxel...")
        return np.random.rand(grid_size, grid_size, grid_size).astype(np.float32)

class ModernVoxelCNN(nn.Module):
    """Modern 3D CNN with batch normalization and dropout"""
    def __init__(self, num_classes=10, voxel_size=32):
        super(ModernVoxelCNN, self).__init__()
        
        # Convolutional layers with batch normalization
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        
        self.pool = nn.MaxPool3d(2)
        self.dropout = nn.Dropout3d(0.3)
        
        # Calculate flattened size
        # After 3 pooling operations: voxel_size -> voxel_size/8
        flattened_size = 128 * (voxel_size // 8) ** 3
        
        self.fc1 = nn.Linear(flattened_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # Convolutional layers
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)
        
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.fc3(x)
        
        return x

def create_sample_data():
    """Create sample voxel data for demonstration"""
    print("Creating sample 3D object data...")
    
    # Create a simple chair-like structure
    voxel_size = 32
    voxel = np.zeros((voxel_size, voxel_size, voxel_size), dtype=np.float32)
    
    # Chair structure
    center = voxel_size // 2
    
    # Seat
    voxel[center-6:center+6, center-2:center+2, center-6:center+6] = 1.0
    
    # Back
    voxel[center-6:center+6, center+2:center+8, center+4:center+6] = 1.0
    
    # Legs
    voxel[center-4:center-2, center-8:center-2, center-4:center-2] = 1.0
    voxel[center+2:center+4, center-8:center-2, center-4:center-2] = 1.0
    voxel[center-4:center-2, center-8:center-2, center+2:center+4] = 1.0
    voxel[center+2:center+4, center-8:center-2, center+2:center+4] = 1.0
    
    return voxel

def main():
    """Main demonstration function"""
    print("🔍 3D Object Recognition System - Original Implementation")
    print("=" * 60)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create sample data
    if os.path.exists('data/ModelNet10/chair/train/chair_0001.off'):
        print("Loading real mesh data...")
        voxel = mesh_to_voxel('data/ModelNet10/chair/train/chair_0001.off')
    else:
        print("Creating synthetic sample data...")
        voxel = create_sample_data()
    
    # Convert to tensor
    vox_tensor = torch.tensor(voxel).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 32, 32, 32)
    
    # Initialize model
    print("Initializing modern 3D CNN...")
    model = ModernVoxelCNN(num_classes=10, voxel_size=32).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Test inference
    print("Running inference...")
    model.eval()
    with torch.no_grad():
        output = model(vox_tensor.to(device))
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(output, dim=1).item()
    
    # Class names
    class_names = [
        'chair', 'table', 'sofa', 'bed', 'desk',
        'bookshelf', 'dresser', 'nightstand', 'wardrobe', 'cabinet'
    ]
    
    print(f"🔍 Predicted class: {class_names[predicted_class]}")
    print(f"📊 Confidence: {probabilities[0][predicted_class]:.3f}")
    print(f"📈 All class probabilities:")
    
    for i, (name, prob) in enumerate(zip(class_names, probabilities[0])):
        print(f"  {i}: {name}: {prob:.3f}")
    
    print("\n✅ Demo completed successfully!")
    print("\nFor the full-featured system with training, UI, and visualization:")
    print("  python modern_3d_recognition.py")
    print("  streamlit run streamlit_app.py")

if __name__ == "__main__":
    main()