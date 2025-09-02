import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys
import os
from PIL import Image
import torch
import torch.nn.functional as F

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

# Safe imports with error handling
try:
    from models.cnn import ResNet18Binary
    from models.wildlife_cnn import WildlifeClassifier
except ImportError:
    st.error("Could not import models. Please check your model implementation.")
    ResNet18Binary = None
    WildlifeClassifier = None

try:
    from services.llm_service import LLMService
except ImportError:
    st.warning("LLM service not available. Some features will be limited.")
    LLMService = None

# Page configuration
st.set_page_config(
    page_title="Animal Classification AI",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
        border-left: 4px solid #1f77b4;
    }
    .explanation-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #1f77b4;
        margin: 1rem 0;
        color: #2c3e50;
        font-size: 16px;
        line-height: 1.6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .ai-response {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .ai-response h3 {
        color: white !important;
        margin-top: 0;
    }
    .ai-response p {
        color: white !important;
        margin-bottom: 10px;
    }
    .upload-section {
        border: 2px dashed #ccc;
        border-radius: 0.5rem;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def get_device():
    """Get the appropriate device for model inference"""
    if torch.cuda.is_available():
        try:
            # Test CUDA availability
            torch.cuda.current_device()
            return "cuda"
        except:
            return "cpu"
    return "cpu"

@st.cache_resource
def load_models():
    """Load both cat/dog and wildlife models with caching"""
    device = get_device()
    models = {}
    
    # Load Cat/Dog model
    try:
        if ResNet18Binary is not None:
            cat_dog_model = ResNet18Binary(pretrained=False)
            classifier_path = Path("checkpoints/week2/resnet18_epoch5.pth")
            if classifier_path.exists():
                cat_dog_model.load_state_dict(torch.load(classifier_path, map_location=device))
                cat_dog_model = cat_dog_model.to(device)
                cat_dog_model.eval()
                models['cat_dog'] = {
                    'model': cat_dog_model,
                    'classes': ['cat', 'dog'],
                    'type': 'binary'
                }
    except Exception as e:
        st.warning(f"Could not load cat/dog model: {e}")
    
    # Load Wildlife model
    try:
        if WildlifeClassifier is not None:
            wildlife_path = Path("checkpoints/wildlife/best_wildlife_model.pth")
            if wildlife_path.exists():
                # Default wildlife classes (will be updated when model loads)
                wildlife_classes = [
                    'bear', 'tiger', 'lion', 'wolf', 'elephant', 
                    'dolphin', 'snake', 'turtle', 'beaver', 'kangaroo'
                ]
                wildlife_classifier = WildlifeClassifier(wildlife_classes, device)
                wildlife_classifier.load_checkpoint(wildlife_path)
                models['wildlife'] = {
                    'model': wildlife_classifier.model,
                    'classifier': wildlife_classifier,
                    'classes': wildlife_classes,
                    'type': 'multiclass'
                }
    except Exception as e:
        st.warning(f"Could not load wildlife model: {e}")
    
    return models, device

def safe_predict(model, image_tensor, device, classes):
    """Safe prediction with comprehensive error handling for deployment"""
    try:
        if model is None:
            # Demo mode - return random prediction
            import random
            predicted_class = random.choice(classes)
            confidence_score = random.uniform(0.7, 0.95)
            probabilities = np.random.dirichlet(np.ones(len(classes)))
            return predicted_class, confidence_score, probabilities
        
        model.eval()
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            predicted_class = classes[predicted.item()]
            confidence_score = confidence.item()
            
            return predicted_class, confidence_score, probabilities[0].cpu().numpy()
            
    except RuntimeError as e:
        if "Input type" in str(e) and "weight type" in str(e):
            # Device mismatch error - force CPU mode
            st.warning("GPU/CPU device mismatch detected. Switching to CPU mode...")
            try:
                model = model.cpu()
                image_tensor = image_tensor.cpu()
                device = "cpu"
                
                with torch.no_grad():
                    outputs = model(image_tensor)
                    probabilities = F.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                    
                    predicted_class = classes[predicted.item()]
                    confidence_score = confidence.item()
                    
                    return predicted_class, confidence_score, probabilities[0].cpu().numpy()
            except Exception as fallback_error:
                st.error(f"Fallback prediction failed: {fallback_error}")
                return "unknown", 0.5, np.array([0.5, 0.5])
        else:
            st.error(f"Prediction error: {e}")
            return "unknown", 0.5, np.array([0.5, 0.5])
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None, None

def check_device_compatibility():
    """Check and display device compatibility information"""
    device = get_device()
    
    with st.expander("🔧 System Information", expanded=False):
        st.write(f"**PyTorch Version**: {torch.__version__}")
        st.write(f"**Device**: {device}")
        
        if torch.cuda.is_available():
            try:
                gpu_name = torch.cuda.get_device_name()
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                st.write(f"**GPU**: {gpu_name}")
                st.write(f"**GPU Memory**: {gpu_memory:.1f} GB")
                st.success("✅ CUDA GPU available for acceleration")
            except:
                st.warning("⚠️ CUDA available but GPU info unavailable")
        else:
            st.info("💻 Using CPU for inference")
    
    return device

@st.cache_data
def load_performance_data():
    """Load performance data from output files"""
    try:
        data = {}
        
        # Load week2 metrics
        week2_path = Path("outputs/week2/metrics.json")
        if week2_path.exists():
            with open(week2_path, 'r') as f:
                data['week2'] = json.load(f)
        
        # Load week3 curves
        week3_path = Path("outputs/week3/curves.json")
        if week3_path.exists():
            with open(week3_path, 'r') as f:
                data['week3'] = json.load(f)
        
        # Load week4 RL curve
        week4_path = Path("outputs/week4/rl_curve.json")
        if week4_path.exists():
            with open(week4_path, 'r') as f:
                data['week4'] = json.load(f)
        
        return data
    except Exception as e:
        st.error(f"Error loading performance data: {e}")
        return {}

def preprocess_image(image, model_type='cat_dog'):
    """Preprocess uploaded image for model input"""
    try:
        import torchvision.transforms as transforms
        
        if model_type == 'wildlife':
            # Wildlife model preprocessing (matches training)
            transform = transforms.Compose([
                transforms.Resize(96),  # Match training size
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
            ])
        else:
            # Cat/Dog model preprocessing (original)
            transform = transforms.Compose([
                transforms.Resize(128),
                transforms.ToTensor(),
                # Simple normalization for cat/dog model
            ])
        
        # Apply transforms
        image_tensor = transform(image)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor
        
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🐾 Animal Classification AI</h1>', unsafe_allow_html=True)
    st.markdown("### Intelligent Image Classification with Active Learning & Reinforcement Learning")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "📊 Model Performance", "🔍 Interactive Classification", "📈 Active Learning Analysis", "🤖 LLM Explanations"]
    )
    
    # Load models and data
    models, device = load_models()
    performance_data = load_performance_data()
    
    if page == "🏠 Home":
        show_home_page(models, device, performance_data)
    elif page == "📊 Model Performance":
        show_performance_page(performance_data)
    elif page == "🔍 Interactive Classification":
        show_classification_page(models, device)
    elif page == "📈 Active Learning Analysis":
        show_active_learning_page(performance_data)
    elif page == "🤖 LLM Explanations":
        show_llm_explanations_page(performance_data)

def show_home_page(models, device, performance_data):
    """Display the home page with overview"""
    st.header("🎯 Project Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        This project demonstrates **Policy Gradient Active Learning** for animal image classification using:
        
        - **Deep Learning**: ResNet18 CNN for image classification
        - **Active Learning**: Intelligent sample selection strategies
        - **Reinforcement Learning**: REINFORCE algorithm for policy optimization
        - **Wildlife Extension**: CIFAR-100 multi-class animal classification
        - **LLM Integration**: AI-powered explanations and insights
        
        ### Key Features:
        ✅ **Data Preprocessing**: 25K images processed and organized  
        ✅ **Model Training**: CNN trained with active learning strategies  
        ✅ **RL Policy**: REINFORCE agent for sample selection  
        ✅ **Wildlife Classification**: 10+ animal species support  
        ✅ **Performance Analysis**: Comprehensive evaluation and comparison  
        ✅ **Interactive UI**: User-friendly interface for model exploration  
        ✅ **LLM Explanations**: Human-understandable insights  
        """)
    
    with col2:
        st.markdown("### 🚀 Quick Stats")
        
        # Model status
        if 'cat_dog' in models:
            st.success("✅ Cat/Dog Model Loaded")
        else:
            st.warning("⚠️ Cat/Dog Model Not Available")
            
        if 'wildlife' in models:
            st.success("✅ Wildlife Model Loaded")
            st.info(f"🐾 {len(models['wildlife']['classes'])} Animal Classes")
        else:
            st.warning("⚠️ Wildlife Model Not Available")
        
        if performance_data:
            st.success(f"✅ {len(performance_data)} Performance Datasets Available")
        else:
            st.warning("⚠️ Performance Data Not Available")
        
        st.info("💡 Upload an image to test the models!")

def show_classification_page(models, device):
    """Display interactive image classification"""
    st.header("🔍 Interactive Image Classification")
    
    # Add device compatibility check
    device = check_device_compatibility()
    
    if not models:
        st.error("No models available. Please ensure models are trained and checkpoints are available.")
        return
    
    # Model selection
    st.subheader("🤖 Select Model")
    available_models = list(models.keys())
    model_names = {
        'cat_dog': '🐱🐶 Cat vs Dog Classifier',
        'wildlife': '🦁🐘 Wildlife Multi-Class Classifier'
    }
    
    selected_model = st.selectbox(
        "Choose classification model:",
        available_models,
        format_func=lambda x: model_names.get(x, x)
    )
    
    if selected_model not in models:
        st.error(f"Selected model '{selected_model}' not available.")
        return
    
    model_info = models[selected_model]
    st.info(f"**Classes**: {', '.join(model_info['classes'])}")

    # File upload
    st.subheader("📤 Upload an Image")
    help_text = {
        'cat_dog': "Upload a cat or dog image to classify",
        'wildlife': "Upload an animal image (bear, tiger, lion, wolf, elephant, dolphin, snake, turtle, beaver, kangaroo)"
    }
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg'],
        help=help_text.get(selected_model, "Upload an animal image to classify")
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Preprocess and predict
        if st.button("🔍 Classify Image", type="primary"):
            with st.spinner("Analyzing image..."):
                # Make prediction based on model type
                if selected_model == 'wildlife' and 'classifier' in model_info:
                    # Wildlife model prediction with proper preprocessing
                    try:
                        wildlife_tensor = preprocess_image(image, model_type='wildlife')
                        predicted_class, confidence, probabilities = model_info['classifier'].predict(wildlife_tensor)
                    except Exception as e:
                        st.error(f"Wildlife prediction error: {e}")
                        return
                else:
                    # Cat/Dog model prediction
                    catdog_tensor = preprocess_image(image, model_type='cat_dog')
                    predicted_class, confidence, probabilities = safe_predict(model_info['model'], catdog_tensor, device, model_info['classes'])
                
                if predicted_class is not None:
                    # Display results
                    st.success(f"🎯 **Prediction: {predicted_class.upper()}**")
                    
                    # Confidence meter
                    st.metric("Confidence", f"{confidence:.2%}")
                    
                    # Probability distribution
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Probability Distribution")
                        classes = [cls.title() for cls in model_info['classes']]
                        
                        # Handle different probability array sizes
                        if len(probabilities) == len(classes):
                            fig = px.bar(
                                x=classes, 
                                y=probabilities,
                                title="Class Probabilities",
                                color=probabilities,
                                color_continuous_scale='Blues'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Probability distribution size mismatch")
                    
                    with col2:
                        st.subheader("Confidence Analysis")
                        if confidence > 0.9:
                            st.success("🟢 High Confidence - Model is very certain about this prediction")
                        elif confidence > 0.7:
                            st.info("🟡 Medium Confidence - Model is reasonably certain")
                        else:
                            st.warning("🟠 Low Confidence - Model is uncertain about this prediction")
                    
                    # Model-specific explanation
                    st.subheader("🤖 AI Explanation")
                    if selected_model == 'wildlife':
                        explanation = f"""
                        The wildlife model predicted **{predicted_class.upper()}** with {confidence:.1%} confidence. 
                        This prediction is based on visual features learned from the CIFAR-100 dataset, 
                        including body shape, texture patterns, and distinctive characteristics 
                        that differentiate various animal species.
                        
                        {'This is a high-confidence prediction.' if confidence > 0.8 else 
                         'This is a moderate-confidence prediction.' if confidence > 0.6 else 
                         'This is a low-confidence prediction - the model is uncertain.'}
                        """
                    else:
                        explanation = f"""
                        The model predicted **{predicted_class.upper()}** with {confidence:.1%} confidence. 
                        This prediction is based on visual features the model learned during training, 
                        such as facial structure, ear shape, and other distinguishing characteristics 
                        between cats and dogs.
                        
                        {'This is a high-confidence prediction.' if confidence > 0.8 else 
                         'This is a moderate-confidence prediction.' if confidence > 0.6 else 
                         'This is a low-confidence prediction - the model is uncertain.'}
                        """
                    st.markdown(f'<div class="explanation-box">{explanation}</div>', unsafe_allow_html=True)
                    
                    # Show top-k predictions for wildlife model
                    if selected_model == 'wildlife' and 'classifier' in model_info:
                        try:
                            top_predictions = model_info['classifier'].get_top_k_predictions(wildlife_tensor, k=3)
                            st.subheader("🏆 Top 3 Predictions")
                            for i, (class_name, prob) in enumerate(top_predictions):
                                st.write(f"{i+1}. **{class_name.title()}**: {prob:.2%}")
                        except Exception as e:
                            st.warning(f"Could not get top predictions: {e}")

def show_performance_page(performance_data):
    """Display comprehensive model performance analysis"""
    st.header("📊 Model Performance Analysis")
    
    if not performance_data:
        st.warning("No performance data available. Please run the training scripts first.")
        return
    
    # Calculate comprehensive metrics
    baseline_acc = max(performance_data.get('week2', {}).get('val_acc', [0]))
    al_data = performance_data.get('week3', {})
    rl_data = performance_data.get('week4', {})
    
    # Performance Summary Cards
    st.subheader("🎯 Performance Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Baseline CNN", 
            f"{baseline_acc:.1%}",
            help="Full dataset performance (20K samples)"
        )
    
    with col2:
        random_best = max(al_data.get('Random', [0])) if 'Random' in al_data else 0
        random_gap = (baseline_acc - random_best) * 100
        st.metric(
            "Random AL", 
            f"{random_best:.1%}",
            f"-{random_gap:.1f}%",
            help="Random sampling active learning"
        )
    
    with col3:
        uncertainty_best = max(al_data.get('Uncertainty', [0])) if 'Uncertainty' in al_data else 0
        uncertainty_gap = (baseline_acc - uncertainty_best) * 100
        st.metric(
            "Uncertainty AL", 
            f"{uncertainty_best:.1%}",
            f"-{uncertainty_gap:.1f}%",
            help="Uncertainty-based active learning"
        )
    
    with col4:
        rl_best = max(rl_data.get('RL', [0])) if 'RL' in rl_data else 0
        rl_gap = (baseline_acc - rl_best) * 100
        st.metric(
            "REINFORCE AL", 
            f"{rl_best:.1%}",
            f"-{rl_gap:.1f}%",
            help="Reinforcement learning active learning"
        )
    
    # Detailed Performance Analysis
    st.subheader("📈 Learning Curves Comparison")
    
    if al_data and rl_data:
        # Create learning curves plot
        fig = go.Figure()
        
        rounds = list(range(len(al_data.get('Random', []))))
        
        # Add traces for each method
        if 'Random' in al_data:
            fig.add_trace(go.Scatter(
                x=rounds, y=al_data['Random'],
                mode='lines+markers', name='Random AL',
                line=dict(color='red', width=3),
                marker=dict(size=8)
            ))
        
        if 'Uncertainty' in al_data:
            fig.add_trace(go.Scatter(
                x=rounds, y=al_data['Uncertainty'],
                mode='lines+markers', name='Uncertainty AL',
                line=dict(color='blue', width=3),
                marker=dict(size=8)
            ))
        
        if 'RL' in rl_data:
            rl_subset = rl_data['RL'][:len(rounds)]
            fig.add_trace(go.Scatter(
                x=rounds, y=rl_subset,
                mode='lines+markers', name='REINFORCE AL',
                line=dict(color='green', width=3),
                marker=dict(size=8)
            ))
        
        # Add baseline reference
        fig.add_hline(
            y=baseline_acc, 
            line_dash="dash", 
            line_color="black",
            annotation_text=f"Baseline CNN ({baseline_acc:.1%})"
        )
        
        fig.update_layout(
            title="Active Learning Performance Comparison",
            xaxis_title="Active Learning Round",
            yaxis_title="Validation Accuracy",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Sample Efficiency Analysis
    st.subheader("💡 Sample Efficiency Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Data efficiency metrics
        st.markdown("**Data Efficiency Metrics:**")
        
        data_reduction = (1 - 5000/20000) * 100
        st.write(f"• **Data Reduction**: {data_reduction:.0f}% (5K vs 20K samples)")
        
        if random_best > 0:
            random_retention = (random_best/baseline_acc) * 100
            st.write(f"• **Random AL**: {random_retention:.1f}% performance retention")
        
        if uncertainty_best > 0:
            uncertainty_retention = (uncertainty_best/baseline_acc) * 100
            st.write(f"• **Uncertainty AL**: {uncertainty_retention:.1f}% performance retention")
        
        if rl_best > 0:
            rl_retention = (rl_best/baseline_acc) * 100
            st.write(f"• **REINFORCE AL**: {rl_retention:.1f}% performance retention")
    
    with col2:
        # Sample efficiency chart
        methods = ['Baseline CNN', 'Random AL', 'Uncertainty AL', 'REINFORCE AL']
        samples = [20000, 5000, 5000, 5000]
        accuracies = [baseline_acc*100, random_best*100, uncertainty_best*100, rl_best*100]
        
        efficiency = [acc/sample*1000 for acc, sample in zip(accuracies, samples)]
        
        fig_eff = go.Figure(data=[
            go.Bar(x=methods, y=efficiency, 
                   text=[f'{e:.1f}' for e in efficiency],
                   textposition='auto',
                   marker_color=['black', 'red', 'blue', 'green'])
        ])
        
        fig_eff.update_layout(
            title="Data Efficiency (Accuracy per 1K Samples)",
            yaxis_title="Accuracy per 1K Samples",
            height=400
        )
        
        st.plotly_chart(fig_eff, use_container_width=True)
    
    # Detailed Training History
    st.subheader("📋 Training History Details")
    
    if 'week2' in performance_data:
        with st.expander("🔍 Baseline CNN Training Details", expanded=False):
            week2_data = performance_data['week2']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Training Loss Progression:**")
                epochs = list(range(1, len(week2_data['train_loss']) + 1))
                
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(
                    x=epochs, y=week2_data['train_loss'],
                    mode='lines+markers', name='Training Loss',
                    line=dict(color='orange', width=3)
                ))
                
                fig_loss.update_layout(
                    title="Training Loss Over Epochs",
                    xaxis_title="Epoch",
                    yaxis_title="Loss",
                    height=300
                )
                
                st.plotly_chart(fig_loss, use_container_width=True)
            
            with col2:
                st.markdown("**Validation Accuracy Progression:**")
                
                fig_acc = go.Figure()
                fig_acc.add_trace(go.Scatter(
                    x=epochs, y=week2_data['val_acc'],
                    mode='lines+markers', name='Validation Accuracy',
                    line=dict(color='green', width=3)
                ))
                
                fig_acc.update_layout(
                    title="Validation Accuracy Over Epochs",
                    xaxis_title="Epoch", 
                    yaxis_title="Accuracy",
                    height=300
                )
                
                st.plotly_chart(fig_acc, use_container_width=True)
    
    # REINFORCE Policy Analysis
    if 'RL' in rl_data:
        with st.expander("🤖 REINFORCE Policy Analysis", expanded=False):
            st.markdown("**Policy Learning Progression:**")
            
            rl_curve = rl_data['RL']
            episodes = list(range(1, len(rl_curve) + 1))
            
            fig_rl = go.Figure()
            fig_rl.add_trace(go.Scatter(
                x=episodes, y=rl_curve,
                mode='lines+markers', name='RL Policy Performance',
                line=dict(color='purple', width=2),
                marker=dict(size=4)
            ))
            
            # Add moving average
            window_size = 5
            if len(rl_curve) >= window_size:
                moving_avg = []
                for i in range(len(rl_curve)):
                    start_idx = max(0, i - window_size + 1)
                    moving_avg.append(np.mean(rl_curve[start_idx:i+1]))
                
                fig_rl.add_trace(go.Scatter(
                    x=episodes, y=moving_avg,
                    mode='lines', name='Moving Average (5 episodes)',
                    line=dict(color='darkred', width=3, dash='dash')
                ))
            
            fig_rl.update_layout(
                title="REINFORCE Policy Learning Curve (45 Episodes)",
                xaxis_title="Episode",
                yaxis_title="Validation Accuracy",
                height=400
            )
            
            st.plotly_chart(fig_rl, use_container_width=True)
            
            # Policy statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Best Episode", f"{max(rl_curve):.1%}")
            
            with col2:
                st.metric("Average Performance", f"{np.mean(rl_curve):.1%}")
            
            with col3:
                st.metric("Performance Variance", f"{np.std(rl_curve):.3f}")
    
    # Performance Insights with LLM
    st.subheader("🧠 AI Performance Insights")
    
    if st.button("Generate AI Analysis", type="primary"):
        with st.spinner("Analyzing performance data..."):
            try:
                # Initialize LLM service with NVIDIA API
                llm_service = LLMService(provider="nvidia")
                
                # Prepare comprehensive results for analysis
                all_results = {
                    'baseline': performance_data.get('week2', {}),
                    'active_learning': performance_data.get('week3', {}),
                    'reinforce': performance_data.get('week4', {})
                }
                
                # Generate comprehensive research summary
                research_response = llm_service.generate_research_summary(all_results)
                
                st.markdown("### 🔬 Research Summary")
                st.markdown(f'<div class="explanation-box">{research_response.text}</div>', 
                           unsafe_allow_html=True)
                
                # Generate academic insights
                method_comparison = {
                    'Baseline_CNN': baseline_acc,
                    'Random_AL': random_best,
                    'Uncertainty_AL': uncertainty_best,
                    'REINFORCE_AL': rl_best
                }
                
                sample_efficiency = {
                    'Baseline_CNN': 20000,
                    'Active_Learning_Methods': 5000
                }
                
                academic_response = llm_service.generate_academic_insights(
                    method_comparison, sample_efficiency
                )
                
                st.markdown("### 🎓 Academic Insights")
                st.markdown(f'<div class="explanation-box">{academic_response.text}</div>', 
                           unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error generating AI analysis: {e}")
                st.info("AI analysis temporarily unavailable. Please check the API configuration.")

def show_active_learning_page(performance_data):
    """Display comprehensive active learning analysis"""
    st.header("📈 Active Learning Analysis")
    
    if not performance_data:
        st.warning("No active learning data available. Please run main_week3.py and main_week4.py first.")
        return
    
    # Extract data
    al_data = performance_data.get('week3', {})
    rl_data = performance_data.get('week4', {})
    baseline_data = performance_data.get('week2', {})
    
    if not al_data and not rl_data:
        st.warning("No active learning results found.")
        return
    
    st.markdown("""
    Active Learning reduces labeling costs by intelligently selecting the most informative samples for annotation.
    This analysis compares different sample selection strategies.
    """)
    
    # Strategy Comparison Overview
    st.subheader("🎯 Strategy Comparison Overview")
    
    strategies_info = {
        "Random Sampling": {
            "description": "Randomly selects samples from the unlabeled pool",
            "pros": ["Simple to implement", "No computational overhead", "Unbiased selection"],
            "cons": ["Inefficient use of labeling budget", "May select redundant samples"],
            "color": "red"
        },
        "Uncertainty Sampling": {
            "description": "Selects samples where the model is most uncertain (highest entropy)",
            "pros": ["Focuses on decision boundary", "Theoretically motivated", "Often effective"],
            "cons": ["May be biased toward outliers", "Doesn't consider diversity"],
            "color": "blue"
        },
        "REINFORCE Policy": {
            "description": "Uses reinforcement learning to learn optimal sample selection policy",
            "pros": ["Learns from experience", "Adapts to data distribution", "Outperforms heuristics"],
            "cons": ["Requires training time", "More complex implementation"],
            "color": "green"
        }
    }
    
    # Display strategy information
    for strategy, info in strategies_info.items():
        with st.expander(f"📋 {strategy}", expanded=False):
            st.write(f"**Description**: {info['description']}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Advantages:**")
                for pro in info['pros']:
                    st.write(f"• {pro}")
            
            with col2:
                st.write("**Limitations:**")
                for con in info['cons']:
                    st.write(f"• {con}")

def show_llm_explanations_page(performance_data):
    """Display LLM-powered explanations"""
    st.header("🤖 AI-Powered Explanations & Insights")
    
    st.markdown("""
    Get AI-powered explanations and insights about your Policy Gradient Active Learning project. 
    This section uses advanced language models to provide human-readable analysis of your results.
    """)
    
    if LLMService is None:
        st.warning("LLM service is not available. Please install the required dependencies.")
        st.info("To enable LLM features, run: `pip install -r requirements.txt`")
        return
    
    # Initialize LLM service
    try:
        llm_service = LLMService(provider="nvidia")  # Using NVIDIA API as configured
        st.success("✅ AI Analysis Service Connected")
    except Exception as e:
        st.error(f"Failed to initialize LLM service: {e}")
        return
    
    # Create tabs for different types of explanations
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔬 Research Summary", 
        "📊 Performance Analysis", 
        "🎯 Strategy Comparison", 
        "💡 Recommendations"
    ])
    
    with tab1:
        st.subheader("🔬 Comprehensive Research Summary")
        st.markdown("Get an AI-generated summary of your entire research project:")
        
        if st.button("Generate Research Summary", type="primary", key="research_summary"):
            with st.spinner("Analyzing your research results..."):
                try:
                    # Prepare comprehensive results
                    all_results = {
                        'baseline': performance_data.get('week2', {}),
                        'active_learning': performance_data.get('week3', {}),
                        'reinforce': performance_data.get('week4', {})
                    }
                    
                    response = llm_service.generate_research_summary(all_results)
                    
                    st.markdown("### 📋 AI Research Analysis")
                    
                    # Display with better formatting
                    st.markdown(f"""
                    <div class="ai-response">
                        <h3>🔬 Research Summary</h3>
                        <p>{response.text}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Also show as regular markdown for better readability
                    with st.container():
                        st.markdown("**AI Analysis:**")
                        st.write(response.text)
                    
                    # Show metadata
                    with st.expander("🔧 Analysis Details", expanded=False):
                        st.json(response.metadata)
                        st.write(f"**Confidence**: {response.confidence:.1%}")
                        
                except Exception as e:
                    st.error(f"Error generating research summary: {e}")
    
    with tab2:
        st.subheader("📊 Performance Analysis")
        st.markdown("Get detailed explanations of your model performance:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Analyze Baseline Performance", key="baseline_perf"):
                with st.spinner("Analyzing baseline CNN performance..."):
                    try:
                        baseline_data = performance_data.get('week2', {})
                        if baseline_data:
                            metrics = {
                                "peak_accuracy": max(baseline_data.get('val_acc', [0])),
                                "final_loss": baseline_data.get('train_loss', [0])[-1] if baseline_data.get('train_loss') else 0,
                                "training_epochs": len(baseline_data.get('val_acc', [])),
                                "dataset_size": 20000
                            }
                            
                            response = llm_service.summarize_model_performance(
                                metrics, 
                                training_history=baseline_data.get('val_acc', [])
                            )
                            
                            st.markdown("### 🎯 Baseline CNN Analysis")
                            
                            # Better formatted display
                            st.success("✅ Analysis Complete!")
                            st.markdown("**AI Explanation:**")
                            st.write(response.text)
                            
                            # Alternative styled display
                            st.info(f"💡 **Insight**: {response.text}")
                        else:
                            st.warning("No baseline performance data available")
                    except Exception as e:
                        st.error(f"Error analyzing baseline performance: {e}")
        
        with col2:
            if st.button("Analyze Active Learning Performance", key="al_perf"):
                with st.spinner("Analyzing active learning results..."):
                    try:
                        al_data = performance_data.get('week3', {})
                        rl_data = performance_data.get('week4', {})
                        
                        if al_data or rl_data:
                            # Calculate comparative metrics
                            random_best = max(al_data.get('Random', [0])) if 'Random' in al_data else 0
                            uncertainty_best = max(al_data.get('Uncertainty', [0])) if 'Uncertainty' in al_data else 0
                            rl_best = max(rl_data.get('RL', [0])) if 'RL' in rl_data else 0
                            
                            metrics = {
                                "random_sampling_peak": random_best,
                                "uncertainty_sampling_peak": uncertainty_best,
                                "reinforce_policy_peak": rl_best,
                                "sample_efficiency": "75% data reduction",
                                "active_learning_rounds": 9
                            }
                            
                            response = llm_service.summarize_model_performance(metrics)
                            
                            st.markdown("### 🎯 Active Learning Analysis")
                            
                            # Clear, visible display
                            st.success("✅ Analysis Complete!")
                            st.markdown("**AI Explanation:**")
                            st.write(response.text)
                            
                            # Additional formatting
                            with st.expander("📊 Detailed Analysis", expanded=True):
                                st.markdown(response.text)
                        else:
                            st.warning("No active learning performance data available")
                    except Exception as e:
                        st.error(f"Error analyzing active learning performance: {e}")
    
    with tab3:
        st.subheader("🎯 Strategy Comparison")
        st.markdown("Understand the differences between active learning strategies:")
        
        strategy_options = ["Random Sampling", "Uncertainty Sampling", "REINFORCE Policy"]
        selected_strategy = st.selectbox("Select strategy to analyze:", strategy_options)
        
        if st.button(f"Explain {selected_strategy}", key="strategy_explain"):
            with st.spinner(f"Analyzing {selected_strategy} strategy..."):
                try:
                    # Prepare performance comparison data
                    al_data = performance_data.get('week3', {})
                    rl_data = performance_data.get('week4', {})
                    
                    performance_comparison = {}
                    if 'Random' in al_data:
                        performance_comparison['Random'] = al_data['Random']
                    if 'Uncertainty' in al_data:
                        performance_comparison['Uncertainty'] = al_data['Uncertainty']
                    if 'RL' in rl_data:
                        performance_comparison['REINFORCE'] = rl_data['RL'][:9]  # Match AL rounds
                    
                    response = llm_service.explain_active_learning_strategy(
                        selected_strategy, 
                        performance_comparison
                    )
                    
                    st.markdown(f"### 📋 {selected_strategy} Strategy Analysis")
                    
                    # Clear display with multiple formats
                    st.success("✅ Strategy Analysis Complete!")
                    
                    # Main explanation
                    st.markdown("**AI Explanation:**")
                    st.write(response.text)
                    
                    # Highlighted version
                    st.info(f"🎯 **Key Insight**: {response.text}")
                    
                except Exception as e:
                    st.error(f"Error explaining strategy: {e}")
    
    with tab4:
        st.subheader("💡 AI Recommendations")
        st.markdown("Get personalized recommendations for improving your model:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Get Improvement Recommendations", type="primary", key="improvements"):
                with st.spinner("Generating improvement recommendations..."):
                    try:
                        # Calculate current best performance
                        all_accuracies = []
                        if performance_data.get('week3', {}).get('Random'):
                            all_accuracies.extend(performance_data['week3']['Random'])
                        if performance_data.get('week3', {}).get('Uncertainty'):
                            all_accuracies.extend(performance_data['week3']['Uncertainty'])
                        if performance_data.get('week4', {}).get('RL'):
                            all_accuracies.extend(performance_data['week4']['RL'])
                        
                        current_accuracy = max(all_accuracies) if all_accuracies else 0.9
                        
                        response = llm_service.recommend_improvements(
                            current_accuracy=current_accuracy,
                            dataset_size=5000,
                            active_learning_rounds=9
                        )
                        
                        st.markdown("### 🚀 Improvement Recommendations")
                        
                        # Clear, actionable display
                        st.success("✅ Recommendations Generated!")
                        
                        # Main recommendations
                        st.markdown("**AI Recommendations:**")
                        st.write(response.text)
                        
                        # Highlighted format
                        st.warning(f"💡 **Action Items**: {response.text}")
                        
                    except Exception as e:
                        st.error(f"Error generating recommendations: {e}")
        
        with col2:
            if st.button("Generate Academic Insights", key="academic"):
                with st.spinner("Generating academic-level insights..."):
                    try:
                        # Prepare method comparison
                        baseline_acc = max(performance_data.get('week2', {}).get('val_acc', [0]))
                        random_acc = max(performance_data.get('week3', {}).get('Random', [0]))
                        uncertainty_acc = max(performance_data.get('week3', {}).get('Uncertainty', [0]))
                        rl_acc = max(performance_data.get('week4', {}).get('RL', [0]))
                        
                        method_comparison = {
                            "Baseline_CNN": baseline_acc,
                            "Random_AL": random_acc,
                            "Uncertainty_AL": uncertainty_acc,
                            "REINFORCE_AL": rl_acc
                        }
                        
                        sample_efficiency = {
                            "Baseline_CNN": 20000,
                            "Active_Learning_Methods": 5000
                        }
                        
                        response = llm_service.generate_academic_insights(
                            method_comparison, 
                            sample_efficiency
                        )
                        
                        st.markdown("### 🎓 Academic Research Insights")
                        
                        # Academic-style display
                        st.success("✅ Academic Analysis Complete!")
                        
                        # Main insights
                        st.markdown("**Research Insights:**")
                        st.write(response.text)
                        
                        # Academic format
                        with st.expander("📚 Full Academic Analysis", expanded=True):
                            st.markdown(response.text)
                            
                        # Summary box
                        st.info(f"🎓 **Academic Summary**: {response.text[:200]}..." if len(response.text) > 200 else response.text)
                        
                    except Exception as e:
                        st.error(f"Error generating academic insights: {e}")
    
    # Add a simple test section
    st.markdown("---")
    st.subheader("🧪 Test AI Response")
    
    if st.button("Test AI Connection", key="test_ai"):
        with st.spinner("Testing AI connection..."):
            try:
                # Simple test prompt
                test_response = llm_service._call_llm("Explain what active learning is in one sentence.")
                
                st.success("✅ AI Connection Working!")
                st.markdown("**Test Response:**")
                st.write(test_response.text)
                
                # Show response details
                with st.expander("🔧 Response Details"):
                    st.write(f"**Text Length**: {len(test_response.text)} characters")
                    st.write(f"**Confidence**: {test_response.confidence}")
                    st.json(test_response.metadata)
                    
            except Exception as e:
                st.error(f"AI Connection Failed: {e}")
                st.info("Using fallback explanations...")
    
    # Add a footer with API information
    st.markdown("---")
    st.markdown("**🔧 AI Analysis powered by NVIDIA API** | Explanations generated using advanced language models")

if __name__ == "__main__":
    main()