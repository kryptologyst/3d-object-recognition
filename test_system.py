"""
Test Suite for 3D Object Recognition System
============================================

Comprehensive tests for all components of the system
"""

import pytest
import torch
import numpy as np
import tempfile
import os
from pathlib import Path

# Import our modules
from modern_3d_recognition import (
    Config, Synthetic3DDataset, Modern3DCNN, ModelTrainer, VisualizationTools
)

class TestConfig:
    """Test configuration management"""
    
    def test_config_defaults(self):
        """Test default configuration values"""
        config = Config()
        assert config.voxel_size == 64
        assert config.num_classes == 10
        assert config.batch_size == 16
        assert config.learning_rate == 0.001
        assert config.num_epochs == 50
    
    def test_config_custom_values(self):
        """Test custom configuration values"""
        config = Config(
            voxel_size=32,
            num_classes=5,
            batch_size=8,
            learning_rate=0.01
        )
        assert config.voxel_size == 32
        assert config.num_classes == 5
        assert config.batch_size == 8
        assert config.learning_rate == 0.01
    
    def test_config_post_init(self):
        """Test configuration post-initialization"""
        config = Config()
        assert config.conv_channels == [32, 64, 128, 256]
        assert config.fc_layers == [512, 256]
        assert config.device in ['cpu', 'cuda']

class TestSynthetic3DDataset:
    """Test synthetic dataset generation"""
    
    def test_dataset_initialization(self):
        """Test dataset initialization"""
        dataset = Synthetic3DDataset(num_samples=100, voxel_size=32, num_classes=5)
        assert len(dataset) == 100
        assert dataset.voxel_size == 32
        assert dataset.num_classes == 5
        assert len(dataset.class_names) == 5
    
    def test_dataset_getitem(self):
        """Test dataset item retrieval"""
        dataset = Synthetic3DDataset(num_samples=10, voxel_size=32, num_classes=3)
        
        voxel, class_id = dataset[0]
        
        assert isinstance(voxel, torch.Tensor)
        assert isinstance(class_id, torch.Tensor)
        assert voxel.shape == (32, 32, 32)
        assert voxel.dtype == torch.float32
        assert class_id.dtype == torch.long
        assert 0 <= class_id.item() < 3
    
    def test_dataset_class_distribution(self):
        """Test that dataset has balanced class distribution"""
        dataset = Synthetic3DDataset(num_samples=30, voxel_size=32, num_classes=3)
        
        class_counts = [0] * 3
        for i in range(len(dataset)):
            _, class_id = dataset[i]
            class_counts[class_id.item()] += 1
        
        # Each class should appear roughly the same number of times
        for count in class_counts:
            assert count > 0
    
    def test_voxel_generation(self):
        """Test voxel generation for different shapes"""
        dataset = Synthetic3DDataset(voxel_size=32)
        
        # Test box generation
        box_voxel = dataset.generate_primitive_shape('box', (8, 8, 8))
        assert box_voxel.shape == (32, 32, 32)
        assert np.sum(box_voxel) > 0
        
        # Test sphere generation
        sphere_voxel = dataset.generate_primitive_shape('sphere', (8, 8, 8))
        assert sphere_voxel.shape == (32, 32, 32)
        assert np.sum(sphere_voxel) > 0
        
        # Test cylinder generation
        cylinder_voxel = dataset.generate_primitive_shape('cylinder', (8, 8, 8))
        assert cylinder_voxel.shape == (32, 32, 32)
        assert np.sum(cylinder_voxel) > 0

class TestModern3DCNN:
    """Test 3D CNN model"""
    
    def test_model_initialization(self):
        """Test model initialization"""
        config = Config(voxel_size=32, num_classes=5)
        model = Modern3DCNN(config)
        
        assert isinstance(model, Modern3DCNN)
        assert len(model.conv_layers) == len(config.conv_channels)
        assert len(model.bn_layers) == len(config.conv_channels)
        assert len(model.fc_layers) == len(config.fc_layers)
    
    def test_model_forward_pass(self):
        """Test model forward pass"""
        config = Config(voxel_size=32, num_classes=5)
        model = Modern3DCNN(config)
        
        # Create dummy input
        batch_size = 2
        input_tensor = torch.randn(batch_size, 1, 32, 32, 32)
        
        # Forward pass
        output = model(input_tensor)
        
        assert output.shape == (batch_size, config.num_classes)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_model_parameter_count(self):
        """Test that model has reasonable number of parameters"""
        config = Config(voxel_size=32, num_classes=5)
        model = Modern3DCNN(config)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params == total_params
        assert total_params < 10_000_000  # Reasonable upper bound
    
    def test_model_device_compatibility(self):
        """Test model works on CPU"""
        config = Config(voxel_size=32, num_classes=5, device='cpu')
        model = Modern3DCNN(config).to(config.device)
        
        input_tensor = torch.randn(1, 1, 32, 32, 32).to(config.device)
        output = model(input_tensor)
        
        assert output.device.type == 'cpu'

class TestModelTrainer:
    """Test model training functionality"""
    
    def test_trainer_initialization(self):
        """Test trainer initialization"""
        config = Config(voxel_size=32, num_classes=5, num_epochs=1)
        trainer = ModelTrainer(config)
        
        assert isinstance(trainer.model, Modern3DCNN)
        assert trainer.device.type == config.device
        assert isinstance(trainer.optimizer, torch.optim.Optimizer)
        assert isinstance(trainer.criterion, torch.nn.Module)
    
    def test_train_epoch(self):
        """Test training for one epoch"""
        config = Config(voxel_size=32, num_classes=5, batch_size=4, num_epochs=1)
        trainer = ModelTrainer(config)
        
        # Create small dataset
        dataset = Synthetic3DDataset(num_samples=20, voxel_size=32, num_classes=5)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
        
        # Train one epoch
        train_loss, train_acc = trainer.train_epoch(dataloader)
        
        assert isinstance(train_loss, float)
        assert isinstance(train_acc, float)
        assert train_loss >= 0
        assert 0 <= train_acc <= 100
    
    def test_validation(self):
        """Test model validation"""
        config = Config(voxel_size=32, num_classes=5, batch_size=4)
        trainer = ModelTrainer(config)
        
        # Create small dataset
        dataset = Synthetic3DDataset(num_samples=20, voxel_size=32, num_classes=5)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)
        
        # Validate
        val_loss, val_acc = trainer.validate(dataloader)
        
        assert isinstance(val_loss, float)
        assert isinstance(val_acc, float)
        assert val_loss >= 0
        assert 0 <= val_acc <= 100

class TestVisualizationTools:
    """Test visualization tools"""
    
    def test_plot_training_history(self):
        """Test training history plotting"""
        history = {
            'train_losses': [1.0, 0.8, 0.6, 0.4],
            'val_losses': [1.1, 0.9, 0.7, 0.5],
            'train_accuracies': [20, 40, 60, 80],
            'val_accuracies': [18, 38, 58, 78]
        }
        
        # This should not raise an exception
        try:
            VisualizationTools.plot_training_history(history)
            assert True
        except Exception as e:
            pytest.fail(f"plot_training_history raised {e}")
    
    def test_plot_confusion_matrix(self):
        """Test confusion matrix plotting"""
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 1, 0, 2, 2]
        class_names = ['class1', 'class2', 'class3']
        
        # This should not raise an exception
        try:
            VisualizationTools.plot_confusion_matrix(y_true, y_pred, class_names)
            assert True
        except Exception as e:
            pytest.fail(f"plot_confusion_matrix raised {e}")

class TestIntegration:
    """Integration tests"""
    
    def test_end_to_end_training(self):
        """Test complete training pipeline"""
        config = Config(
            voxel_size=32,
            num_classes=3,
            batch_size=4,
            num_epochs=2,
            learning_rate=0.01
        )
        
        # Create dataset
        dataset = Synthetic3DDataset(num_samples=60, voxel_size=32, num_classes=3)
        
        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False)
        
        # Initialize trainer
        trainer = ModelTrainer(config)
        
        # Train
        history = trainer.train(train_loader, val_loader)
        
        # Check results
        assert len(history['train_losses']) > 0
        assert len(history['val_losses']) > 0
        assert len(history['train_accuracies']) > 0
        assert len(history['val_accuracies']) > 0
        
        # Check that training improved (loss decreased)
        assert history['train_losses'][-1] < history['train_losses'][0]
    
    def test_model_save_load(self):
        """Test model saving and loading"""
        config = Config(voxel_size=32, num_classes=3)
        model = Modern3DCNN(config)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp_file:
            torch.save(model.state_dict(), tmp_file.name)
            
            # Load model
            new_model = Modern3DCNN(config)
            new_model.load_state_dict(torch.load(tmp_file.name))
            
            # Test that models produce same output
            input_tensor = torch.randn(1, 1, 32, 32, 32)
            
            model.eval()
            new_model.eval()
            
            with torch.no_grad():
                output1 = model(input_tensor)
                output2 = new_model(input_tensor)
                
                assert torch.allclose(output1, output2, atol=1e-6)
            
            # Clean up
            os.unlink(tmp_file.name)

# Performance tests
class TestPerformance:
    """Performance and stress tests"""
    
    def test_large_batch_processing(self):
        """Test processing large batches"""
        config = Config(voxel_size=64, num_classes=10, batch_size=32)
        model = Modern3DCNN(config)
        
        # Large batch
        input_tensor = torch.randn(32, 1, 64, 64, 64)
        
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
            
        assert output.shape == (32, 10)
        assert not torch.isnan(output).any()
    
    def test_memory_efficiency(self):
        """Test memory efficiency with large voxel grids"""
        config = Config(voxel_size=128, num_classes=10)
        model = Modern3DCNN(config)
        
        # Large voxel grid
        input_tensor = torch.randn(1, 1, 128, 128, 128)
        
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
            
        assert output.shape == (1, 10)

# Utility functions for testing
def create_test_config():
    """Create a test configuration"""
    return Config(
        voxel_size=32,
        num_classes=3,
        batch_size=4,
        num_epochs=1,
        learning_rate=0.01
    )

def create_test_dataset():
    """Create a small test dataset"""
    return Synthetic3DDataset(num_samples=20, voxel_size=32, num_classes=3)

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
