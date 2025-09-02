"""
Enhanced Dashboard Components for Policy Gradient Active Learning
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch
from PIL import Image

class ExperimentDashboard:
    def __init__(self):
        self.results_path = Path("outputs")
        self.checkpoints_path = Path("checkpoints")
        
    def load_all_results(self):
        """Load results from all weeks"""
        results = {}
        
        # Week 1 - Preprocessing
        week1_path = self.results_path / "week1" / "preprocess_stats.json"
        if week1_path.exists():
            with open(week1_path) as f:
                results['preprocessing'] = json.load(f)
        
        # Week 2 - Baseline
        week2_path = self.results_path / "week2" / "metrics.json"
        if week2_path.exists():
            with open(week2_path) as f:
                results['baseline'] = json.load(f)
        
        # Week 3 - Active Learning Baselines
        week3_path = self.results_path / "week3" / "curves.json"
        if week3_path.exists():
            with open(week3_path) as f:
                results['active_learning'] = json.load(f)
        
        # Week 4 - REINFORCE
        week4_path = self.results_path / "week4" / "rl_curve.json"
        if week4_path.exists():
            with open(week4_path) as f:
                results['reinforce'] = json.load(f)
        
        return results
    
    def create_training_curves_comparison(self, results):
        """Create comprehensive training curves comparison"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Baseline Training Loss', 'Baseline Validation Accuracy', 
                          'Active Learning Comparison', 'REINFORCE Policy Learning'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Baseline Training Loss
        if 'baseline' in results:
            epochs = list(range(1, len(results['baseline']['train_loss']) + 1))
            fig.add_trace(
                go.Scatter(x=epochs, y=results['baseline']['train_loss'],
                          name='Training Loss', line=dict(color='red')),
                row=1, col=1
            )
        
        # Baseline Validation Accuracy
        if 'baseline' in results:
            epochs = list(range(1, len(results['baseline']['val_acc']) + 1))
            fig.add_trace(
                go.Scatter(x=epochs, y=results['baseline']['val_acc'],
                          name='Validation Accuracy', line=dict(color='blue')),
                row=1, col=2
            )
        
        # Active Learning Comparison
        if 'active_learning' in results:
            for method, accuracies in results['active_learning'].items():
                rounds = list(range(len(accuracies)))
                labeled_samples = [1000 + r * 500 for r in rounds]
                fig.add_trace(
                    go.Scatter(x=labeled_samples, y=accuracies,
                              name=f'{method} Sampling', mode='lines+markers'),
                    row=2, col=1
                )
        
        # REINFORCE Policy Learning
        if 'reinforce' in results:
            steps = list(range(len(results['reinforce']['RL'])))
            fig.add_trace(
                go.Scatter(x=steps, y=results['reinforce']['RL'],
                          name='REINFORCE Policy', line=dict(color='purple')),
                row=2, col=2
            )
        
        fig.update_layout(height=800, showlegend=True, 
                         title_text="Comprehensive Training Analysis")
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_xaxes(title_text="Labeled Samples", row=2, col=1)
        fig.update_xaxes(title_text="Training Step", row=2, col=2)
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy", row=1, col=2)
        fig.update_yaxes(title_text="Accuracy", row=2, col=1)
        fig.update_yaxes(title_text="Accuracy", row=2, col=2)
        
        return fig
    
    def create_performance_summary_table(self, results):
        """Create performance summary table"""
        summary_data = []
        
        # Baseline performance
        if 'baseline' in results:
            baseline_peak = max(results['baseline']['val_acc'])
            summary_data.append({
                'Method': 'Baseline CNN',
                'Peak Accuracy': f"{baseline_peak:.4f}",
                'Sample Efficiency': 'Full Dataset',
                'Training Time': '~30 min',
                'Use Case': 'Maximum Accuracy'
            })
        
        # Active Learning methods
        if 'active_learning' in results:
            for method, accuracies in results['active_learning'].items():
                peak_acc = max(accuracies)
                samples_90 = None
                for i, acc in enumerate(accuracies):
                    if acc >= 0.90:
                        samples_90 = 1000 + i * 500
                        break
                
                summary_data.append({
                    'Method': f'{method} Sampling',
                    'Peak Accuracy': f"{peak_acc:.4f}",
                    'Sample Efficiency': f'{samples_90} samples' if samples_90 else 'Not reached',
                    'Training Time': '~2-3 hours',
                    'Use Case': 'Consistent AL' if method == 'Uncertainty' else 'Simple Baseline'
                })
        
        # REINFORCE
        if 'reinforce' in results:
            rl_peak = max(results['reinforce']['RL'])
            summary_data.append({
                'Method': 'REINFORCE Policy',
                'Peak Accuracy': f"{rl_peak:.4f}",
                'Sample Efficiency': 'Adaptive',
                'Training Time': '~2.5 hours',
                'Use Case': 'Intelligent AL'
            })
        
        return pd.DataFrame(summary_data)
    
    def create_sample_efficiency_plot(self, results):
        """Create sample efficiency comparison"""
        fig = go.Figure()
        
        if 'active_learning' in results:
            for method, accuracies in results['active_learning'].items():
                rounds = list(range(len(accuracies)))
                labeled_samples = [1000 + r * 500 for r in rounds]
                
                fig.add_trace(go.Scatter(
                    x=labeled_samples, 
                    y=accuracies,
                    mode='lines+markers',
                    name=f'{method} Sampling',
                    line=dict(width=3)
                ))
        
        # Add 90% accuracy line
        fig.add_hline(y=0.90, line_dash="dash", line_color="red",
                     annotation_text="90% Accuracy Target")
        
        fig.update_layout(
            title="Sample Efficiency Comparison",
            xaxis_title="Number of Labeled Samples",
            yaxis_title="Validation Accuracy",
            height=500,
            showlegend=True
        )
        
        return fig
    
    def create_confusion_matrix_heatmap(self, y_true, y_pred, method_name):
        """Create confusion matrix heatmap"""
        cm = confusion_matrix(y_true, y_pred)
        
        fig = px.imshow(cm, 
                       labels=dict(x="Predicted", y="Actual", color="Count"),
                       x=['Cat', 'Dog'], 
                       y=['Cat', 'Dog'],
                       color_continuous_scale='Blues',
                       title=f'Confusion Matrix - {method_name}')
        
        # Add text annotations
        for i in range(len(cm)):
            for j in range(len(cm[0])):
                fig.add_annotation(x=j, y=i, text=str(cm[i][j]),
                                 showarrow=False, font=dict(color="white" if cm[i][j] > cm.max()/2 else "black"))
        
        return fig
    
    def display_queried_samples(self, sample_indices, dataset_path, method_name):
        """Display samples selected by different methods"""
        st.subheader(f"Samples Selected by {method_name}")
        
        # Create columns for sample display
        cols = st.columns(5)
        
        for i, idx in enumerate(sample_indices[:10]):  # Show first 10 samples
            col = cols[i % 5]
            
            # This would need actual implementation to load images
            # For now, show placeholder
            with col:
                st.write(f"Sample {idx}")
                # st.image(image_path, caption=f"Index: {idx}")
    
    def generate_insights_summary(self, results):
        """Generate automated insights from results"""
        insights = []
        
        if 'baseline' in results and 'active_learning' in results:
            baseline_peak = max(results['baseline']['val_acc'])
            al_peaks = {method: max(accs) for method, accs in results['active_learning'].items()}
            best_al_method = max(al_peaks, key=al_peaks.get)
            best_al_acc = al_peaks[best_al_method]
            
            efficiency = (best_al_acc / baseline_peak) * 100
            insights.append(f"🎯 **Sample Efficiency**: {best_al_method} achieved {efficiency:.1f}% of baseline performance with only 20% of the data")
        
        if 'reinforce' in results and 'active_learning' in results:
            rl_peak = max(results['reinforce']['RL'])
            uncertainty_peak = max(results['active_learning'].get('Uncertainty', [0]))
            
            if rl_peak > uncertainty_peak:
                improvement = ((rl_peak - uncertainty_peak) / uncertainty_peak) * 100
                insights.append(f"🚀 **REINFORCE Advantage**: Policy learning outperformed uncertainty sampling by {improvement:.2f}%")
            
        if 'active_learning' in results:
            methods_90 = {}
            for method, accuracies in results['active_learning'].items():
                for i, acc in enumerate(accuracies):
                    if acc >= 0.90:
                        methods_90[method] = 1000 + i * 500
                        break
            
            if methods_90:
                best_efficiency = min(methods_90.values())
                best_method = min(methods_90, key=methods_90.get)
                insights.append(f"⚡ **Fastest to 90%**: {best_method} reached 90% accuracy with only {best_efficiency} labeled samples")
        
        return insights

def render_enhanced_dashboard():
    """Render the enhanced experiment dashboard"""
    st.title("🔬 Policy Gradient Active Learning - Experiment Dashboard")
    
    dashboard = ExperimentDashboard()
    results = dashboard.load_all_results()
    
    if not results:
        st.warning("No experimental results found. Please run the experiments first.")
        return
    
    # Sidebar for navigation
    st.sidebar.title("Dashboard Navigation")
    page = st.sidebar.selectbox("Choose Analysis", [
        "📊 Overview & Summary",
        "📈 Training Curves",
        "🎯 Sample Efficiency",
        "🔍 Detailed Analysis",
        "🤖 AI Insights"
    ])
    
    if page == "📊 Overview & Summary":
        st.header("Experiment Overview")
        
        # Performance summary table
        summary_df = dashboard.create_performance_summary_table(results)
        st.subheader("Performance Summary")
        st.dataframe(summary_df, use_container_width=True)
        
        # Key metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        if 'preprocessing' in results:
            with col1:
                st.metric("Total Images", f"{results['preprocessing']['train_count'] + results['preprocessing']['val_count']:,}")
        
        if 'baseline' in results:
            with col2:
                baseline_peak = max(results['baseline']['val_acc'])
                st.metric("Baseline Peak", f"{baseline_peak:.2%}")
        
        if 'active_learning' in results:
            with col3:
                al_peaks = [max(accs) for accs in results['active_learning'].values()]
                best_al = max(al_peaks) if al_peaks else 0
                st.metric("Best AL Method", f"{best_al:.2%}")
        
        if 'reinforce' in results:
            with col4:
                rl_peak = max(results['reinforce']['RL'])
                st.metric("REINFORCE Peak", f"{rl_peak:.2%}")
    
    elif page == "📈 Training Curves":
        st.header("Comprehensive Training Analysis")
        
        # Main comparison plot
        comparison_fig = dashboard.create_training_curves_comparison(results)
        st.plotly_chart(comparison_fig, use_container_width=True)
        
        # Individual method analysis
        st.subheader("Method-Specific Analysis")
        
        if 'baseline' in results:
            st.write("**Baseline CNN Training**")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"Final Training Loss: {results['baseline']['train_loss'][-1]:.4f}")
                st.write(f"Loss Reduction: {((results['baseline']['train_loss'][0] - results['baseline']['train_loss'][-1]) / results['baseline']['train_loss'][0] * 100):.1f}%")
            with col2:
                st.write(f"Final Validation Accuracy: {results['baseline']['val_acc'][-1]:.2%}")
                st.write(f"Peak Validation Accuracy: {max(results['baseline']['val_acc']):.2%}")
    
    elif page == "🎯 Sample Efficiency":
        st.header("Sample Efficiency Analysis")
        
        # Sample efficiency plot
        efficiency_fig = dashboard.create_sample_efficiency_plot(results)
        st.plotly_chart(efficiency_fig, use_container_width=True)
        
        # Efficiency metrics
        if 'active_learning' in results:
            st.subheader("Efficiency Metrics")
            
            efficiency_data = []
            for method, accuracies in results['active_learning'].items():
                samples_90 = None
                samples_95 = None
                
                for i, acc in enumerate(accuracies):
                    if acc >= 0.90 and samples_90 is None:
                        samples_90 = 1000 + i * 500
                    if acc >= 0.95 and samples_95 is None:
                        samples_95 = 1000 + i * 500
                
                efficiency_data.append({
                    'Method': method,
                    'Samples to 90%': samples_90 or 'Not reached',
                    'Samples to 95%': samples_95 or 'Not reached',
                    'Peak Accuracy': f"{max(accuracies):.2%}",
                    'Final Accuracy': f"{accuracies[-1]:.2%}"
                })
            
            efficiency_df = pd.DataFrame(efficiency_data)
            st.dataframe(efficiency_df, use_container_width=True)
    
    elif page == "🔍 Detailed Analysis":
        st.header("Detailed Method Analysis")
        
        # Method selection
        available_methods = []
        if 'active_learning' in results:
            available_methods.extend(results['active_learning'].keys())
        if 'reinforce' in results:
            available_methods.append('REINFORCE')
        
        selected_method = st.selectbox("Select Method for Analysis", available_methods)
        
        if selected_method in results.get('active_learning', {}):
            accuracies = results['active_learning'][selected_method]
            
            # Detailed statistics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Peak Accuracy", f"{max(accuracies):.2%}")
                st.metric("Final Accuracy", f"{accuracies[-1]:.2%}")
                st.metric("Mean Accuracy", f"{np.mean(accuracies):.2%}")
            
            with col2:
                st.metric("Std Deviation", f"{np.std(accuracies):.4f}")
                st.metric("Improvement", f"{((accuracies[-1] - accuracies[0]) / accuracies[0] * 100):.1f}%")
                st.metric("Consistency", f"{(1 - np.std(accuracies)/np.mean(accuracies)):.2%}")
        
        elif selected_method == 'REINFORCE':
            accuracies = results['reinforce']['RL']
            
            # REINFORCE-specific analysis
            st.subheader("REINFORCE Policy Analysis")
            
            # Policy learning phases
            policy_epochs = 5
            steps_per_epoch = len(accuracies) // policy_epochs
            
            epoch_performance = []
            for epoch in range(policy_epochs):
                start_idx = epoch * steps_per_epoch
                end_idx = (epoch + 1) * steps_per_epoch
                epoch_accs = accuracies[start_idx:end_idx]
                epoch_performance.append({
                    'Epoch': epoch + 1,
                    'Mean Accuracy': np.mean(epoch_accs),
                    'Peak Accuracy': max(epoch_accs),
                    'Std Deviation': np.std(epoch_accs)
                })
            
            epoch_df = pd.DataFrame(epoch_performance)
            st.dataframe(epoch_df, use_container_width=True)
    
    elif page == "🤖 AI Insights":
        st.header("Automated Insights & Recommendations")
        
        # Generate insights
        insights = dashboard.generate_insights_summary(results)
        
        st.subheader("Key Findings")
        for insight in insights:
            st.write(insight)
        
        # Recommendations
        st.subheader("Recommendations for Improvement")
        
        recommendations = [
            "🔧 **Hyperparameter Tuning**: Experiment with different learning rates for 1-2% accuracy gains",
            "📊 **Data Augmentation**: Add rotation, scaling, and color jittering for improved robustness",
            "🎯 **Ensemble Methods**: Combine multiple models for higher accuracy and reliability",
            "⚡ **Early Stopping**: Implement early stopping to prevent overfitting and reduce training time",
            "🔄 **Cross-Validation**: Use k-fold CV for more robust performance estimates"
        ]
        
        for rec in recommendations:
            st.write(rec)
        
        # Future work suggestions
        st.subheader("Future Research Directions")
        
        future_work = [
            "🌍 **Multi-Class Extension**: Extend to wildlife datasets with 10+ animal classes",
            "🧠 **Advanced RL**: Experiment with PPO, A3C, or other policy gradient methods",
            "🔍 **Interpretability**: Add attention mechanisms to understand what the policy learns",
            "📱 **Deployment**: Create mobile app for real-time animal classification",
            "🏥 **Domain Transfer**: Apply to medical imaging or other critical domains"
        ]
        
        for work in future_work:
            st.write(work)