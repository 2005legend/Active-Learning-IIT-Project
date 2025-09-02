# Policy Gradient Active Learning for Animal Image Classification

**Reducing labeling effort using Reinforcement Learning-based Active Learning strategies for animal image datasets.**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-green.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Overview

This project addresses the critical challenge of **expensive data labeling** in computer vision by implementing a novel approach that combines **Active Learning** with **Reinforcement Learning** for intelligent sample selection. Traditional supervised learning requires massive labeled datasets, but our approach achieves comparable accuracy using only **7.5% of labeled data**.

### Why This Matters
- **Wildlife Research**: Reduce annotation costs for species classification
- **Medical Imaging**: Minimize expert labeling requirements
- **Industrial Applications**: Scale AI deployment with limited labeled data
- **Research Impact**: Demonstrate practical RL applications in data-efficient learning

### Our Approach
1. **Baseline CNN**: ResNet18 for animal image classification
2. **Active Learning**: Smart sample selection strategies (Random, Uncertainty, REINFORCE)
3. **Policy Gradient RL**: REINFORCE algorithm learns optimal selection policy
4. **LLM Integration**: Human-interpretable explanations via OpenAI API
5. **Interactive UI**: Streamlit-based demonstration platform

### Final Deliverable
- **Trained Models**: CNN classifiers with 91% accuracy on cats vs dogs
- **RL Policy**: REINFORCE agent achieving 92.5% sample efficiency
- **Visualizations**: Comprehensive performance analysis and comparisons
- **Interactive UI**: Real-time classification with confidence analysis

---

## 🎯 Objectives

1. **Reduce Labeling Cost**: Achieve high accuracy with minimal labeled samples
2. **Compare AL Strategies**: Benchmark Random vs Uncertainty vs RL-based selection
3. **Implement RL Agent**: REINFORCE policy gradient for sample selection
4. **Provide Interpretability**: LLM-powered explanations for model decisions
5. **Build Interactive Demo**: User-friendly interface for model exploration

---

## 📊 Dataset

### Primary Development Dataset
- **Source**: [Cats vs Dogs (Kaggle)](https://www.kaggle.com/c/dogs-vs-cats)
- **Size**: 25,000 training images, 12,500 test images
- **Classes**: Binary classification (Cat, Dog)
- **Resolution**: Variable (resized to 128×128 for training)

### Extension Dataset
- **Source**: CIFAR-100 Wildlife Subset
- **Classes**: 10 animal species (bear, tiger, lion, wolf, elephant, dolphin, snake, turtle, beaver, kangaroo)
- **Size**: 5,000 training, 1,000 test images
- **Resolution**: 32×32 (upscaled to 96×96)

### Preprocessing Pipeline
```python
# Training transforms
transforms.Compose([
    transforms.Resize(128),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
```

### Data Splits
- **Training**: 80% (20,000 images)
- **Validation**: 15% (3,750 images)  
- **Test**: 5% (1,250 images)

---

## 🔬 Methodology

### Stage 1: Baseline Supervised Model
**Objective**: Establish performance benchmark with full dataset

- **Architecture**: ResNet18 with pretrained ImageNet weights
- **Training**: 5 epochs, Adam optimizer (lr=0.001)
- **Results**: 91% validation accuracy
- **Implementation**: `main_week2.py`

```python
# Model architecture
model = ResNet18Binary(pretrained=True)
model.fc = nn.Linear(512, 2)  # Binary classification
```

### Stage 2: Active Learning Strategies
**Objective**: Compare intelligent sample selection methods

#### Random Sampling (Baseline)
- Randomly select samples for labeling
- No intelligence in selection process
- Performance ceiling due to inefficient sampling

#### Uncertainty Sampling
- Select samples with highest prediction uncertainty
- Metrics: Entropy, Least Confidence, Margin sampling
- Focus on decision boundary samples

```python
# Entropy-based uncertainty
def entropy_sampling(predictions, k):
    entropy = -torch.sum(predictions * torch.log(predictions + 1e-8), dim=1)
    _, indices = torch.topk(entropy, k)
    return indices
```

#### Comparison Results
- **Random**: 85% accuracy with 2,000 samples
- **Uncertainty**: 89% accuracy with 1,500 samples  
- **Improvement**: 26% reduction in labeling requirements

### Stage 3: Reinforcement Learning (Policy Gradient)
**Objective**: Learn optimal sample selection policy

#### Environment Design
- **State**: CNN feature representations (512-dim)
- **Action**: Binary decision (select/reject sample)
- **Reward**: Validation accuracy improvement
- **Policy**: Multi-layer perceptron (MLP)

```python
# REINFORCE Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=512, hidden_dims=[256, 128, 64]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 2))  # Select/Reject
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return F.softmax(self.network(x), dim=-1)
```

#### REINFORCE Algorithm
1. **Episode Generation**: Policy selects samples for labeling
2. **Reward Calculation**: Train CNN, measure validation accuracy
3. **Policy Update**: Gradient ascent on expected reward
4. **Exploration**: ε-greedy strategy for sample diversity

#### Training Process
- **Episodes**: 50 training episodes
- **Batch Size**: 100 samples per episode
- **Learning Rate**: 0.001 (policy), 0.01 (CNN)
- **Reward Design**: `reward = (new_accuracy - baseline_accuracy) * 100`

### Stage 4: LLM Integration
**Objective**: Provide human-interpretable explanations

#### Implementation
- **API**: OpenAI GPT-3.5-turbo
- **Input**: Image features + prediction confidence
- **Output**: Natural language explanation
- **Context**: Model architecture and training details

```python
# LLM explanation generation
def generate_explanation(prediction, confidence, features):
    prompt = f"""
    Explain this animal classification result:
    - Prediction: {prediction}
    - Confidence: {confidence:.2%}
    - Key features: {features}
    
    Provide a clear, educational explanation.
    """
    return openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
```

### Stage 5: Interactive UI
**Objective**: Demonstrate system capabilities

#### Streamlit Application Features
- **Image Upload**: Drag-and-drop interface
- **Real-time Classification**: Instant predictions
- **Confidence Analysis**: Visual confidence indicators
- **Strategy Comparison**: Side-by-side AL method comparison
- **Performance Visualization**: Interactive charts and metrics
- **LLM Explanations**: Natural language insights

---

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Unlabeled     │    │   RL Policy      │    │   CNN Model     │
│   Image Pool    │───▶│   (REINFORCE)    │───▶│   (ResNet18)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         │                        ▼                        ▼
         │              ┌──────────────────┐    ┌─────────────────┐
         │              │  Sample Selection│    │   Predictions   │
         │              │   (Top-K)        │    │  + Confidence   │
         │              └──────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Human         │    │   Training       │    │   LLM           │
│   Annotation    │◀───│   Loop           │    │   Explanations  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Folder Structure
```
animal_al_rl/
├── data/                   # Raw and processed datasets
│   ├── raw/               # Original dataset files
│   ├── processed/         # Preprocessed images
│   └── splits/            # Train/val/test splits
├── src/                   # Core source code
│   ├── models/            # Neural network architectures
│   │   ├── cnn.py        # ResNet18 classifier
│   │   └── wildlife_cnn.py # Multi-class extension
│   ├── active_learning/   # AL strategy implementations
│   │   ├── strategies.py  # Random, Uncertainty sampling
│   │   └── rl_policy.py   # REINFORCE policy
│   ├── environment/       # RL environment
│   │   └── al_env.py     # Custom Gymnasium environment
│   ├── services/          # External services
│   │   └── llm_service.py # OpenAI API integration
│   ├── utils/             # Utility functions
│   │   ├── data_utils.py  # Data loading and preprocessing
│   │   └── metrics.py     # Evaluation metrics
│   └── visualization/     # Plotting and analysis
│       └── dashboard.py   # Streamlit components
├── checkpoints/           # Saved model weights
│   ├── week2/            # Baseline CNN models
│   └── week4/            # RL policy checkpoints
├── outputs/               # Experimental results
│   ├── figures/          # Generated plots
│   ├── metrics/          # Performance logs
│   └── logs/             # Training logs
├── config/                # Configuration files
│   └── config.yaml       # Hyperparameters
├── notebooks/             # Jupyter notebooks for analysis
├── app.py                 # Streamlit UI application
├── main_week2.py         # Baseline training script
├── main_week3.py         # Active learning experiments
├── main_week4.py         # RL policy training
├── run_experiments.py    # Automated experiment runner
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## 🧪 Experiments & Results

### Training Performance

#### Baseline CNN (Week 2)
- **Final Accuracy**: 91.2%
- **Training Time**: 45 minutes (5 epochs)
- **Convergence**: Stable after epoch 3

![Training Curves](outputs/figures/baseline_training.png)

#### Active Learning Comparison (Week 3)
| Strategy | Samples Used | Final Accuracy | Efficiency Gain |
|----------|-------------|----------------|-----------------|
| Random | 2,000 | 85.3% | Baseline |
| Uncertainty | 1,500 | 89.1% | 25% fewer samples |
| REINFORCE | 1,500 | 92.5% | 25% fewer samples, +3.4% accuracy |

![Active Learning Comparison](outputs/figures/al_comparison.png)

#### RL Policy Learning (Week 4)
- **Training Episodes**: 50
- **Peak Performance**: 92.5% accuracy
- **Sample Efficiency**: 92.5% reduction vs full dataset
- **Convergence**: Stable policy after 35 episodes

![RL Learning Curve](outputs/figures/rl_learning_curve.png)

### Detailed Analysis

#### Confusion Matrix (Best Model)
```
              Predicted
              Cat   Dog
Actual Cat    456    44   (91.2% recall)
       Dog     31   469   (93.8% recall)

Overall Accuracy: 92.5%
Precision: Cat (93.6%), Dog (91.4%)
```

#### Sample Selection Analysis
- **RL Policy Preferences**: Focuses on boundary cases and diverse samples
- **Uncertainty Sampling**: Concentrates on decision boundary
- **Random Sampling**: No intelligent pattern

#### Wildlife Extension Results
- **Dataset**: CIFAR-100 animal subset (10 classes)
- **Accuracy**: 83% with enhanced training
- **Top Performers**: Dolphin (95%), Tiger (94%), Snake (93%)
- **Challenging Classes**: Bear (67%), requires more data

---

## 🖥️ UI Demo

### Streamlit Application Demo video


#### video: https://youtu.be/_Qqjjs3Cvw0


#### Example Workflow
1. **Upload Image**: Drag and drop animal image
2. **Get Prediction**: Instant classification with confidence
3. **View Analysis**: Probability distribution and confidence level
4. **Read Explanation**: LLM-generated natural language explanation

### Key Features
- **Real-time Classification**: Sub-second inference
- **Confidence Visualization**: Color-coded confidence levels
- **Interactive Charts**: Plotly-based performance metrics
- **Mobile Responsive**: Works on all devices
- **Error Handling**: Graceful fallbacks for edge cases

---

## 🚀 How to Run

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional, but recommended)
- OpenAI API key (for LLM features)

### Installation
```bash
# Clone repository
git clone https://github.com/your-username/animal_al_rl.git
cd animal_al_rl

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

### Training Pipeline

#### 1. Baseline Model Training
```bash
python main_week2.py --config config/baseline.yaml
```

#### 2. Active Learning Experiments
```bash
python main_week3.py --strategy uncertainty --samples 1500
python main_week3.py --strategy random --samples 2000
```

#### 3. RL Policy Training
```bash
python main_week4.py --episodes 50 --batch_size 100
```

#### 4. Run All Experiments
```bash
python run_experiments.py --full_pipeline
```

### Interactive UI
```bash
streamlit run app.py
```
Navigate to `http://localhost:8501` to access the interface.

### Configuration
Edit `config/config.yaml` to modify hyperparameters:
```yaml
model:
  architecture: "resnet18"
  pretrained: true
  num_classes: 2

training:
  epochs: 5
  batch_size: 32
  learning_rate: 0.001
  
active_learning:
  initial_samples: 500
  query_size: 100
  max_iterations: 15

rl_policy:
  hidden_dims: [256, 128, 64]
  learning_rate: 0.001
  episodes: 50
```

---

## 🔧 Enhancements & Extensions

### Implemented Extensions
1. **Multi-class Wildlife Dataset**: CIFAR-100 animal subset with 10 species
2. **Enhanced Data Augmentation**: Advanced preprocessing pipeline
3. **Interactive Visualizations**: Real-time performance monitoring
4. **LLM Integration**: Natural language explanations
5. **Mobile-Responsive UI**: Cross-platform compatibility

### Potential Future Enhancements
1. **Advanced RL Algorithms**: PPO, A2C, SAC for improved sample selection
2. **Vision Transformers**: Replace CNN backbone with ViT
3. **Semi-Supervised Learning**: Leverage unlabeled data more effectively
4. **Federated Learning**: Distributed training across institutions
5. **Real-time Deployment**: Edge device optimization
6. **Multi-modal Learning**: Incorporate text and audio data

### Scalability Considerations
- **Larger Datasets**: Tested up to 100K images
- **More Classes**: Extensible to 100+ animal species
- **Cloud Deployment**: Ready for AWS/GCP deployment
- **Batch Processing**: Support for bulk image classification

---

## ⚠️ Limitations

### Technical Limitations
1. **Computational Requirements**: RL training is compute-intensive (4-6 hours on GPU)
2. **API Dependencies**: LLM explanations require stable internet connection
3. **Dataset Size**: Current experiments limited to moderate-scale datasets
4. **Memory Usage**: Full dataset loading requires 8GB+ RAM

### Methodological Limitations
1. **Domain Specificity**: Trained specifically on animal images
2. **Binary Focus**: Primary development on binary classification
3. **Reward Design**: Simple accuracy-based rewards may not capture all aspects
4. **Exploration**: Limited exploration strategies in RL policy

### Practical Limitations
1. **Annotation Quality**: Assumes high-quality human labels
2. **Class Imbalance**: Performance may degrade with severely imbalanced datasets
3. **Real-time Constraints**: Current setup not optimized for real-time applications
4. **Interpretability**: RL policy decisions can be difficult to interpret

---

## 🔮 Future Work

### Research Directions
1. **Advanced RL Methods**
   - Implement PPO and A2C algorithms
   - Multi-agent active learning systems
   - Hierarchical reinforcement learning

2. **Self-Supervised Learning**
   - Incorporate contrastive learning (SimCLR, MoCo)
   - Masked image modeling (MAE, BEiT)
   - Combine with active learning strategies

3. **Real-World Applications**
   - Wildlife camera trap datasets
   - Medical imaging (X-rays, MRIs)
   - Industrial quality control

4. **Theoretical Analysis**
   - Sample complexity bounds
   - Convergence guarantees for RL policies
   - Optimal stopping criteria

### Technical Improvements
1. **Model Architecture**
   - Vision Transformers (ViT, Swin)
   - EfficientNet variants
   - Neural Architecture Search (NAS)

2. **Training Efficiency**
   - Mixed precision training
   - Gradient checkpointing
   - Model parallelism

3. **Deployment Optimization**
   - Model quantization
   - ONNX conversion
   - TensorRT optimization

### Application Extensions
1. **Multi-Domain Learning**: Extend to other domains (text, audio, video)
2. **Few-Shot Learning**: Rapid adaptation to new species
3. **Continual Learning**: Learn new classes without forgetting
4. **Federated Active Learning**: Privacy-preserving distributed learning

---

## 📚 References

### Key Papers
1. **Active Learning**
   - Settles, B. (2009). "Active Learning Literature Survey"
   - Wang, K. et al. (2016). "Cost-Effective Active Learning for Deep Image Classification"

2. **Reinforcement Learning**
   - Williams, R.J. (1992). "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning"
   - Sutton, R.S. & Barto, A.G. (2018). "Reinforcement Learning: An Introduction"

3. **Policy Gradient Methods**
   - Schulman, J. et al. (2017). "Proximal Policy Optimization Algorithms"
   - Mnih, V. et al. (2016). "Asynchronous Methods for Deep Reinforcement Learning"

4. **Active Learning + RL**
   - Bachman, P. et al. (2017). "Learning Algorithms for Active Learning"
   - Pang, K. et al. (2018). "Meta-Learning for Low-Resource Neural Machine Translation"

### Datasets
- [Dogs vs. Cats - Kaggle Competition](https://www.kaggle.com/c/dogs-vs-cats)
- [CIFAR-100 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [ImageNet Large Scale Visual Recognition Challenge](https://image-net.org/challenges/LSVRC/)

### Frameworks & Tools
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Gymnasium](https://gymnasium.farama.org/) - RL environment standard
- [Streamlit](https://streamlit.io/) - Web app framework
- [OpenAI API](https://openai.com/api/) - LLM integration
- [Plotly](https://plotly.com/) - Interactive visualizations

### Additional Resources
- [Deep Learning Book - Goodfellow, Bengio, Courville](https://www.deeplearningbook.org/)
- [Spinning Up in Deep RL - OpenAI](https://spinningup.openai.com/)
- [Active Learning Tutorial - Settles](http://active-learning.net/)

---

## 🙏 Acknowledgements

### Academic Support
- **IIT Minor Project Faculty Panel** for guidance and evaluation
- **Computer Science Department** for computational resources
- **Research Supervisors** for technical mentorship

### Technical Resources
- **OpenAI** for GPT API access and LLM integration
- **Kaggle** for providing the Dogs vs. Cats dataset
- **CIFAR Team** for the CIFAR-100 dataset
- **PyTorch Team** for the excellent deep learning framework

### Open Source Community
- **Streamlit Team** for the intuitive web app framework
- **Gymnasium Developers** for standardized RL environments
- **Plotly Team** for interactive visualization tools
- **GitHub Community** for code hosting and collaboration

### Special Thanks
- **Beta Testers** who provided valuable feedback on the UI
- **Code Reviewers** who helped improve code quality
- **Documentation Contributors** for improving clarity

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

- **Author**: Sidaarth.K
- **Email**: sidaarth005@gmail.com
- **GitHub**: https://github.com/2005legend
- **LinkedIn**: [https://linkedin.com/in/your-profile](https://linkedin.com/in/your-profile)

---

**⭐ If you found this project helpful, please consider giving it a star!**

*This project demonstrates the practical application of reinforcement learning in active learning scenarios, contributing to more efficient and cost-effective machine learning solutions.*
