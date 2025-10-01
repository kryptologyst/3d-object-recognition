"""
Streamlit Web UI for 3D Object Recognition
===========================================

Interactive web interface for uploading and classifying 3D objects
"""

import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import tempfile
import os
from pathlib import Path
import json

# Import our model classes
from modern_3d_recognition import Modern3DCNN, Config, Synthetic3DDataset, VisualizationTools

# Page configuration
st.set_page_config(
    page_title="3D Object Recognition System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 0.5rem;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_model():
    """Load the trained model"""
    config = Config()
    model = Modern3DCNN(config)
    
    # Try to load trained weights
    if os.path.exists('best_model.pth'):
        model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
        return model, config, True
    else:
        return model, config, False

def generate_sample_3d_object(class_name: str, voxel_size: int = 64):
    """Generate a sample 3D object for demonstration"""
    dataset = Synthetic3DDataset(voxel_size=voxel_size)
    class_names = dataset.class_names
    
    if class_name not in class_names:
        class_name = class_names[0]
    
    class_id = class_names.index(class_name)
    
    # Generate a sample
    voxel, _ = dataset[class_id]
    return voxel.numpy(), class_name

def visualize_3d_voxel(voxel, title="3D Object"):
    """Create 3D visualization of voxel grid"""
    # Get coordinates where voxel > 0.5
    x, y, z = np.where(voxel > 0.5)
    
    if len(x) == 0:
        st.warning("No voxels found in the object")
        return None
    
    # Create 3D scatter plot
    fig = go.Figure(data=go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=2,
            color=voxel[x, y, z],
            colorscale='Viridis',
            opacity=0.8,
            colorbar=dict(title="Voxel Value")
        ),
        name='Voxels'
    ))
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube'
        ),
        width=600,
        height=500
    )
    
    return fig

def create_2d_slices(voxel):
    """Create 2D slice visualizations"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['XY Slice (Z=center)', 'XZ Slice (Y=center)', 
                       'YZ Slice (X=center)', '3D Projection'],
        specs=[[{'type': 'heatmap'}, {'type': 'heatmap'}],
               [{'type': 'heatmap'}, {'type': 'scatter3d'}]]
    )
    
    center_z = voxel.shape[2] // 2
    center_y = voxel.shape[1] // 2
    center_x = voxel.shape[0] // 2
    
    # XY slice
    fig.add_trace(
        go.Heatmap(z=voxel[:, :, center_z], colorscale='Viridis'),
        row=1, col=1
    )
    
    # XZ slice
    fig.add_trace(
        go.Heatmap(z=voxel[:, center_y, :], colorscale='Viridis'),
        row=1, col=2
    )
    
    # YZ slice
    fig.add_trace(
        go.Heatmap(z=voxel[center_x, :, :], colorscale='Viridis'),
        row=2, col=1
    )
    
    # 3D projection
    x, y, z = np.where(voxel > 0.5)
    fig.add_trace(
        go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=1)),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False)
    return fig

def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<h1 class="main-header">🔍 3D Object Recognition System</h1>', 
                unsafe_allow_html=True)
    
    # Load model
    model, config, model_loaded = load_model()
    
    # Sidebar
    st.sidebar.title("🎛️ Controls")
    
    # Model status
    if model_loaded:
        st.sidebar.success("✅ Trained model loaded")
    else:
        st.sidebar.warning("⚠️ Using untrained model")
        st.sidebar.info("Train the model first by running the main script")
    
    # Configuration
    st.sidebar.subheader("Configuration")
    voxel_size = st.sidebar.slider("Voxel Size", 32, 128, config.voxel_size, 16)
    num_classes = st.sidebar.slider("Number of Classes", 5, 20, config.num_classes)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Classification", "📊 Visualization", "🎲 Generate Samples", "📈 Model Info"])
    
    with tab1:
        st.header("3D Object Classification")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Upload 3D Object")
            
            # File upload
            uploaded_file = st.file_uploader(
                "Choose a 3D file (.obj, .ply, .stl)",
                type=['obj', 'ply', 'stl'],
                help="Upload a 3D object file for classification"
            )
            
            if uploaded_file is not None:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    # Convert to voxel (simplified - in real implementation, use proper mesh processing)
                    st.info("Converting 3D object to voxel grid...")
                    
                    # For demo purposes, generate a random voxel
                    voxel = np.random.rand(voxel_size, voxel_size, voxel_size)
                    voxel = (voxel > 0.7).astype(np.float32)
                    
                    st.success("✅ Object converted successfully!")
                    
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
                    voxel = None
                finally:
                    # Clean up temp file
                    os.unlink(tmp_path)
            else:
                st.info("Upload a 3D object file to get started")
                voxel = None
        
        with col2:
            st.subheader("Classification Results")
            
            if voxel is not None:
                # Prepare input tensor
                input_tensor = torch.tensor(voxel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                
                # Make prediction
                model.eval()
                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.softmax(output, dim=1)
                    predicted_class = torch.argmax(output, dim=1).item()
                
                # Display results
                class_names = [
                    'chair', 'table', 'sofa', 'bed', 'desk',
                    'bookshelf', 'dresser', 'nightstand', 'wardrobe', 'cabinet'
                ]
                
                st.metric("Predicted Class", class_names[predicted_class])
                
                # Confidence scores
                st.subheader("Confidence Scores")
                confidences = probabilities[0].numpy()
                
                for i, (class_name, conf) in enumerate(zip(class_names, confidences)):
                    st.progress(conf, text=f"{class_name}: {conf:.3f}")
                
                # Top 3 predictions
                top3_indices = torch.topk(probabilities[0], 3).indices
                st.subheader("Top 3 Predictions")
                
                for i, idx in enumerate(top3_indices):
                    st.write(f"{i+1}. {class_names[idx.item()]}: {probabilities[0][idx]:.3f}")
    
    with tab2:
        st.header("3D Visualization")
        
        if voxel is not None:
            # 3D visualization
            st.subheader("3D Voxel Visualization")
            fig_3d = visualize_3d_voxel(voxel, "Uploaded 3D Object")
            if fig_3d:
                st.plotly_chart(fig_3d, use_container_width=True)
            
            # 2D slices
            st.subheader("2D Slices and Projections")
            fig_slices = create_2d_slices(voxel)
            st.plotly_chart(fig_slices, use_container_width=True)
            
            # Statistics
            st.subheader("Object Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Voxels", f"{np.sum(voxel > 0.5):,}")
            with col2:
                st.metric("Volume", f"{np.sum(voxel > 0.5) / (voxel_size**3) * 100:.1f}%")
            with col3:
                st.metric("Dimensions", f"{voxel_size}×{voxel_size}×{voxel_size}")
        else:
            st.info("Upload a 3D object to see visualizations")
    
    with tab3:
        st.header("Generate Sample 3D Objects")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Object Generator")
            
            class_names = [
                'chair', 'table', 'sofa', 'bed', 'desk',
                'bookshelf', 'dresser', 'nightstand', 'wardrobe', 'cabinet'
            ]
            
            selected_class = st.selectbox("Select Object Class", class_names)
            
            if st.button("Generate Sample"):
                sample_voxel, class_name = generate_sample_3d_object(selected_class, voxel_size)
                st.session_state['sample_voxel'] = sample_voxel
                st.session_state['sample_class'] = class_name
        
        with col2:
            st.subheader("Generated Object")
            
            if 'sample_voxel' in st.session_state:
                sample_voxel = st.session_state['sample_voxel']
                sample_class = st.session_state['sample_class']
                
                st.write(f"Generated: **{sample_class}**")
                
                # Visualize generated object
                fig_gen = visualize_3d_voxel(sample_voxel, f"Generated {sample_class}")
                if fig_gen:
                    st.plotly_chart(fig_gen, use_container_width=True)
                
                # Classify generated object
                if st.button("Classify Generated Object"):
                    input_tensor = torch.tensor(sample_voxel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    
                    model.eval()
                    with torch.no_grad():
                        output = model(input_tensor)
                        probabilities = torch.softmax(output, dim=1)
                        predicted_class = torch.argmax(output, dim=1).item()
                    
                    predicted_name = class_names[predicted_class]
                    confidence = probabilities[0][predicted_class].item()
                    
                    if predicted_name == sample_class:
                        st.success(f"✅ Correct! Predicted: {predicted_name} (confidence: {confidence:.3f})")
                    else:
                        st.error(f"❌ Incorrect. Predicted: {predicted_name}, Actual: {sample_class}")
    
    with tab4:
        st.header("Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Architecture")
            st.code(f"""
Voxel Size: {config.voxel_size}
Number of Classes: {config.num_classes}
Convolutional Channels: {config.conv_channels}
Fully Connected Layers: {config.fc_layers}
Dropout Rate: {config.dropout_rate}
            """)
            
            st.subheader("Training Configuration")
            st.code(f"""
Batch Size: {config.batch_size}
Learning Rate: {config.learning_rate}
Number of Epochs: {config.num_epochs}
Device: {config.device}
            """)
        
        with col2:
            st.subheader("Model Statistics")
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            st.metric("Total Parameters", f"{total_params:,}")
            st.metric("Trainable Parameters", f"{trainable_params:,}")
            
            # Model size estimation
            param_size = total_params * 4  # Assuming float32
            st.metric("Estimated Model Size", f"{param_size / 1024 / 1024:.2f} MB")
            
            st.subheader("Class Labels")
            class_names = [
                'chair', 'table', 'sofa', 'bed', 'desk',
                'bookshelf', 'dresser', 'nightstand', 'wardrobe', 'cabinet'
            ]
            
            for i, name in enumerate(class_names):
                st.write(f"{i}: {name}")

if __name__ == "__main__":
    main()
