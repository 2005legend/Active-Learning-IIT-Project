import os
import requests
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class LLMResponse:
    """Structured response from LLM API calls"""
    text: str
    confidence: float
    metadata: Dict[str, Any]

class LLMService:
    """Service for integrating with open-source LLMs through APIs"""
    
    def __init__(self, provider: str = "huggingface", api_key: Optional[str] = None):
        """
        Initialize LLM service
        
        Args:
            provider: LLM provider ('huggingface', 'ollama', 'openai')
            api_key: API key for the provider
        """
        self.provider = provider
        self.api_key = api_key or os.getenv(f"{provider.upper()}_API_KEY")
        
        # Provider-specific configurations
        self.configs = {
            "huggingface": {
                "base_url": "https://api-inference.huggingface.co/models",
                "model": "microsoft/DialoGPT-medium",  # Default model
                "headers": {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            },
            "ollama": {
                "base_url": "http://localhost:11434/api",
                "model": "llama2:7b",  # Default model
                "headers": {}
            },
            "openai": {
                "base_url": "https://api.openai.com/v1",
                "model": "gpt-3.5-turbo",
                "headers": {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            },
            "nvidia": {
                "base_url": "https://integrate.api.nvidia.com/v1",
                "model": "meta/llama-4-maverick-17b-128e-instruct",
                "headers": {"Authorization": "Bearer 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",
                           "Accept": "application/json"}
            }
        }
        
        self.config = self.configs.get(provider, self.configs["huggingface"])
    
    def explain_classification(self, 
                             prediction: str, 
                             confidence: float, 
                             features: Optional[List[float]] = None,
                             image_path: Optional[str] = None) -> LLMResponse:
        """
        Generate human-understandable explanation for classification result
        
        Args:
            prediction: Model prediction (e.g., 'cat', 'dog')
            confidence: Prediction confidence score
            features: Optional feature vector for detailed analysis
            image_path: Optional path to input image
            
        Returns:
            LLMResponse with explanation text
        """
        prompt = self._build_classification_prompt(prediction, confidence, features, image_path)
        return self._call_llm(prompt)
    
    def summarize_model_performance(self, 
                                  metrics: Dict[str, float], 
                                  training_history: Optional[List[float]] = None) -> LLMResponse:
        """
        Generate summary of model performance metrics
        
        Args:
            metrics: Dictionary of performance metrics
            training_history: Optional list of training accuracy/loss values
            
        Returns:
            LLMResponse with performance summary
        """
        prompt = self._build_performance_prompt(metrics, training_history)
        return self._call_llm(prompt)
    
    def recommend_improvements(self, 
                             current_accuracy: float, 
                             dataset_size: int,
                             active_learning_rounds: int) -> LLMResponse:
        """
        Generate recommendations for model improvement
        
        Args:
            current_accuracy: Current model accuracy
            dataset_size: Size of labeled dataset
            active_learning_rounds: Number of active learning rounds completed
            
        Returns:
            LLMResponse with improvement recommendations
        """
        prompt = self._build_improvement_prompt(current_accuracy, dataset_size, active_learning_rounds)
        return self._call_llm(prompt)
    
    def explain_active_learning_strategy(self, 
                                       strategy: str, 
                                       performance_comparison: Dict[str, List[float]]) -> LLMResponse:
        """
        Explain active learning strategy and compare with baselines
        
        Args:
            strategy: Active learning strategy name
            performance_comparison: Dictionary comparing different strategies
            
        Returns:
            LLMResponse with strategy explanation
        """
        prompt = self._build_strategy_prompt(strategy, performance_comparison)
        return self._call_llm(prompt)
    
    def generate_research_summary(self, all_results: dict) -> LLMResponse:
        """
        Generate comprehensive research summary comparing all methods
        
        Args:
            all_results: Dictionary containing results from all experimental phases
            
        Returns:
            LLMResponse with comprehensive research analysis
        """
        prompt = self._build_research_summary_prompt(all_results)
        return self._call_llm(prompt)
    
    def generate_academic_insights(self, 
                                 method_comparison: Dict[str, float],
                                 sample_efficiency: Dict[str, int]) -> LLMResponse:
        """
        Generate academic-level insights for research presentation
        
        Args:
            method_comparison: Peak accuracy comparison between methods
            sample_efficiency: Samples needed to reach target accuracy
            
        Returns:
            LLMResponse with academic insights
        """
        prompt = self._build_academic_insights_prompt(method_comparison, sample_efficiency)
        return self._call_llm(prompt)
    
    def _build_classification_prompt(self, 
                                   prediction: str, 
                                   confidence: float, 
                                   features: Optional[List[float]] = None,
                                   image_path: Optional[str] = None) -> str:
        """Build prompt for classification explanation"""
        prompt = f"""
        You are an AI expert explaining image classification results. 
        
        Prediction: {prediction}
        Confidence: {confidence:.2%}
        
        Please provide a clear, non-technical explanation of this result that would be understandable to a general audience. 
        Include:
        1. What the model predicted and how confident it is
        2. What this means in simple terms
        3. Any factors that might affect the confidence
        4. A brief explanation of how the model makes this decision
        
        Keep the explanation under 150 words and use simple language.
        """
        
        if features:
            prompt += f"\n\nTechnical note: The model analyzed {len(features)} image features to make this decision."
        
        if image_path:
            prompt += f"\n\nImage analyzed: {image_path}"
        
        return prompt
    
    def _build_performance_prompt(self, 
                                metrics: Dict[str, float], 
                                training_history: Optional[List[float]] = None) -> str:
        """Build prompt for performance summary"""
        prompt = f"""
        You are an AI expert summarizing model performance results.
        
        Performance Metrics:
        {json.dumps(metrics, indent=2)}
        
        Please provide a clear summary of these results that would be understandable to stakeholders. 
        Include:
        1. Overall performance assessment
        2. Key strengths and areas of concern
        3. What these numbers mean in practical terms
        4. Whether the model meets typical performance expectations
        
        Keep the summary under 200 words and focus on actionable insights.
        """
        
        if training_history:
            prompt += f"\n\nTraining Progress: The model was trained over {len(training_history)} rounds with varying performance."
        
        return prompt
    
    def _build_improvement_prompt(self, 
                                current_accuracy: float, 
                                dataset_size: int,
                                active_learning_rounds: int) -> str:
        """Build prompt for improvement recommendations"""
        prompt = f"""
        You are an AI expert providing recommendations for model improvement.
        
        Current Situation:
        - Model Accuracy: {current_accuracy:.2%}
        - Labeled Dataset Size: {dataset_size:,} images
        - Active Learning Rounds: {active_learning_rounds}
        
        Please provide specific, actionable recommendations for improving this image classification model. 
        Consider:
        1. Data-related improvements (more samples, data augmentation, etc.)
        2. Model architecture adjustments
        3. Training strategy optimizations
        4. Active learning refinements
        
        Provide 3-5 specific recommendations with brief explanations of expected impact.
        Keep recommendations practical and implementable.
        """
        return prompt
    
    def _build_strategy_prompt(self, 
                             strategy: str, 
                             performance_comparison: Dict[str, List[float]]) -> str:
        """Build prompt for strategy explanation"""
        prompt = f"""
        You are an AI expert explaining active learning strategies and their performance.
        
        Strategy: {strategy}
        Performance Comparison:
        {json.dumps(performance_comparison, indent=2)}
        
        Please explain:
        1. How the {strategy} strategy works in simple terms
        2. How it compares to other approaches
        3. When this strategy would be most effective
        4. Key advantages and limitations
        
        Focus on making the technical concepts accessible to non-experts while providing valuable insights.
        Keep the explanation under 250 words.
        """
        return prompt
    
    def _build_research_summary_prompt(self, all_results: dict) -> str:
        """Build prompt for comprehensive research summary"""
        
        # Extract key metrics
        baseline_peak = max(all_results.get('baseline', {}).get('val_acc', [0]))
        
        al_results = all_results.get('active_learning', {})
        random_peak = max(al_results.get('Random', [0])) if 'Random' in al_results else 0
        uncertainty_peak = max(al_results.get('Uncertainty', [0])) if 'Uncertainty' in al_results else 0
        
        rl_results = all_results.get('reinforce', {})
        rl_peak = max(rl_results.get('RL', [0])) if 'RL' in rl_results else 0
        
        prompt = f"""
        You are a research analyst providing a comprehensive summary of a Policy Gradient Active Learning study.

        EXPERIMENTAL RESULTS:
        - Baseline CNN (Full Dataset): {baseline_peak:.2%} peak accuracy
        - Random Sampling (Active Learning): {random_peak:.2%} peak accuracy  
        - Uncertainty Sampling (Active Learning): {uncertainty_peak:.2%} peak accuracy
        - REINFORCE Policy (RL-based AL): {rl_peak:.2%} peak accuracy

        RESEARCH CONTEXT:
        - Dataset: Dogs vs Cats classification (25,000 images)
        - Active Learning: Started with 1,000 labeled samples, acquired 500 per round
        - REINFORCE: Trained policy network to select most informative samples
        - Objective: Minimize labeling cost while maximizing classification performance

        Please provide a comprehensive research summary including:
        1. **Key Scientific Findings**: What does this study demonstrate about RL-based active learning?
        2. **Method Comparison**: How do the approaches compare in sample efficiency and performance?
        3. **REINFORCE Analysis**: Advantages and limitations of the policy gradient approach
        4. **Practical Implications**: Real-world applications and benefits
        5. **Research Contribution**: How this advances active learning and RL literature

        Write in an academic tone suitable for a research presentation. Keep under 400 words.
        """
        
        return prompt
    
    def _build_academic_insights_prompt(self, 
                                      method_comparison: Dict[str, float],
                                      sample_efficiency: Dict[str, int]) -> str:
        """Build prompt for academic-level insights"""
        
        prompt = f"""
        You are an academic researcher analyzing Policy Gradient Active Learning results.

        METHOD PERFORMANCE COMPARISON:
        {json.dumps(method_comparison, indent=2)}

        SAMPLE EFFICIENCY (samples to reach 90% accuracy):
        {json.dumps(sample_efficiency, indent=2)}

        Please provide academic-level insights focusing on:
        1. **Statistical Significance**: Are the performance differences meaningful?
        2. **Sample Complexity**: How do methods compare in terms of labeled data requirements?
        3. **Theoretical Implications**: What do these results suggest about RL-based active learning?
        4. **Methodological Contributions**: Novel aspects of this approach
        5. **Limitations & Assumptions**: What are the study's constraints?
        6. **Future Research**: Promising directions for follow-up work

        Write for an academic audience familiar with machine learning concepts.
        Focus on rigorous analysis and research implications. Keep under 350 words.
        """
        
        return prompt
    
    def _call_llm(self, prompt: str) -> LLMResponse:
        """Make API call to the selected LLM provider"""
        try:
            if self.provider == "huggingface":
                return self._call_huggingface(prompt)
            elif self.provider == "ollama":
                return self._call_ollama(prompt)
            elif self.provider == "openai":
                return self._call_openai(prompt)
            elif self.provider == "nvidia":
                return self._call_nvidia(prompt)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
        except Exception as e:
            logger.error(f"Error calling LLM API: {e}")
            # Fallback to template-based response
            return self._fallback_response(prompt)
    
    def _call_huggingface(self, prompt: str) -> LLMResponse:
        """Call HuggingFace Inference API"""
        url = f"{self.config['base_url']}/{self.config['model']}"
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 200,
                "temperature": 0.7,
                "do_sample": True
            }
        }
        
        response = requests.post(url, headers=self.config['headers'], json=payload)
        response.raise_for_status()
        
        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            text = result[0].get('generated_text', '')
        else:
            text = str(result)
        
        return LLMResponse(
            text=text,
            confidence=0.8,  # Default confidence for API responses
            metadata={"provider": "huggingface", "model": self.config['model']}
        )
    
    def _call_ollama(self, prompt: str) -> LLMResponse:
        """Call local Ollama API"""
        url = f"{self.config['base_url']}/generate"
        
        payload = {
            "model": self.config['model'],
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 200
            }
        }
        
        response = requests.post(url, headers=self.config['headers'], json=payload)
        response.raise_for_status()
        
        result = response.json()
        text = result.get('response', '')
        
        return LLMResponse(
            text=text,
            confidence=0.8,
            metadata={"provider": "ollama", "model": self.config['model']}
        )
    
    def _call_openai(self, prompt: str) -> LLMResponse:
        """Call OpenAI API"""
        url = f"{self.config['base_url']}/chat/completions"
        
        payload = {
            "model": self.config['model'],
            "messages": [
                {"role": "system", "content": "You are a helpful AI expert that explains technical concepts clearly."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 200,
            "temperature": 0.7
        }
        
        response = requests.post(url, headers=self.config['headers'], json=payload)
        response.raise_for_status()
        
        result = response.json()
        text = result['choices'][0]['message']['content']
        
        return LLMResponse(
            text=text,
            confidence=0.8,
            metadata={"provider": "openai", "model": self.config['model']}
        )
    
    def _call_nvidia(self, prompt: str) -> LLMResponse:
        """Call NVIDIA API"""
        url = f"{self.config['base_url']}/chat/completions"
        
        payload = {
            "model": self.config['model'],
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 512,
            "temperature": 0.7,
            "top_p": 1.00,
            "frequency_penalty": 0.00,
            "presence_penalty": 0.00,
            "stream": False
        }
        
        response = requests.post(url, headers=self.config['headers'], json=payload)
        response.raise_for_status()
        
        result = response.json()
        text = result['choices'][0]['message']['content']
        
        return LLMResponse(
            text=text,
            confidence=0.9,  # NVIDIA models typically high quality
            metadata={"provider": "nvidia", "model": self.config['model']}
        )
    
    def _fallback_response(self, prompt: str) -> LLMResponse:
        """Generate fallback response when API calls fail"""
        if "classification" in prompt.lower():
            text = "The model analyzed the image and made a prediction based on learned patterns. The confidence score indicates how certain the model is about this result."
        elif "performance" in prompt.lower():
            text = "The model has been evaluated on various metrics. These results indicate how well the model performs on the given task."
        elif "improvement" in prompt.lower():
            text = "To improve the model, consider collecting more diverse training data, adjusting hyperparameters, or using different architectures."
        elif "strategy" in prompt.lower():
            text = "Active learning strategies help select the most informative samples for labeling, improving model performance with fewer labeled examples."
        else:
            text = "The model has completed its analysis. Please refer to the technical documentation for detailed information."
        
        return LLMResponse(
            text=text,
            confidence=0.5,
            metadata={"provider": "fallback", "note": "API call failed, using template response"}
        )

# Convenience functions for common use cases
def explain_prediction(prediction: str, confidence: float, **kwargs) -> str:
    """Quick function to explain a single prediction"""
    service = LLMService()
    response = service.explain_classification(prediction, confidence, **kwargs)
    return response.text

def summarize_performance(metrics: Dict[str, float], **kwargs) -> str:
    """Quick function to summarize performance metrics"""
    service = LLMService()
    response = service.summarize_model_performance(metrics, **kwargs)
    return response.text 