#!/usr/bin/env python3
"""
Demo script for testing LLM integration functionality
"""

import sys
from pathlib import Path
import json

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from services.llm_service import LLMService, explain_prediction, summarize_performance

def test_llm_service():
    """Test the LLM service functionality"""
    print("🤖 Testing LLM Service Integration")
    print("=" * 50)
    
    # Test different providers
    providers = ["huggingface", "ollama", "openai"]
    
    for provider in providers:
        print(f"\n🔧 Testing {provider.upper()} provider:")
        try:
            # Initialize service
            service = LLMService(provider=provider)
            print(f"  ✅ Service initialized successfully")
            
            # Test classification explanation
            print(f"  📝 Testing classification explanation...")
            response = service.explain_classification(
                prediction="cat",
                confidence=0.92,
                features=[0.1, 0.8, 0.3, 0.9]
            )
            print(f"  ✅ Explanation generated: {response.text[:100]}...")
            
            # Test performance summary
            print(f"  📊 Testing performance summary...")
            metrics = {
                "validation_accuracy": 0.9378,
                "training_loss": 0.15,
                "test_accuracy": 0.9256
            }
            response = service.summarize_model_performance(metrics)
            print(f"  ✅ Summary generated: {response.text[:100]}...")
            
        except Exception as e:
            print(f"  ❌ Error with {provider}: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Testing Quick Functions")
    
    # Test quick functions
    try:
        explanation = explain_prediction("dog", 0.87)
        print(f"✅ Quick explanation: {explanation[:100]}...")
        
        summary = summarize_performance({"accuracy": 0.94, "loss": 0.12})
        print(f"✅ Quick summary: {summary[:100]}...")
        
    except Exception as e:
        print(f"❌ Error with quick functions: {e}")

def test_with_sample_data():
    """Test LLM service with sample project data"""
    print("\n📊 Testing with Sample Project Data")
    print("=" * 50)
    
    try:
        service = LLMService(provider="huggingface")
        
        # Sample performance data
        sample_metrics = {
            "final_validation_accuracy": 0.9378,
            "peak_rl_performance": 0.9378,
            "training_epochs": 5,
            "active_learning_rounds": 8,
            "dataset_size": 25000
        }
        
        print("📈 Generating performance summary...")
        response = service.summarize_model_performance(sample_metrics)
        print(f"✅ Summary: {response.text}")
        
        print("\n💡 Generating improvement recommendations...")
        response = service.recommend_improvements(
            current_accuracy=0.9378,
            dataset_size=25000,
            active_learning_rounds=8
        )
        print(f"✅ Recommendations: {response.text}")
        
        print("\n🔍 Explaining active learning strategy...")
        sample_comparison = {
            "Random": [0.89, 0.91, 0.93, 0.92],
            "Uncertainty": [0.88, 0.91, 0.93, 0.94],
            "REINFORCE": [0.88, 0.90, 0.93, 0.94]
        }
        response = service.explain_active_learning_strategy("REINFORCE", sample_comparison)
        print(f"✅ Strategy explanation: {response.text}")
        
    except Exception as e:
        print(f"❌ Error testing with sample data: {e}")

def main():
    """Main demo function"""
    print("🚀 Animal Classification AI - LLM Integration Demo")
    print("=" * 60)
    
    # Test basic LLM service
    test_llm_service()
    
    # Test with sample data
    test_with_sample_data()
    
    print("\n" + "=" * 60)
    print("🎉 Demo completed!")
    print("\nTo run the full interactive UI:")
    print("  streamlit run app.py")
    print("\nTo test specific LLM functionality:")
    print("  python -c \"from src.services.llm_service import LLMService; print('LLM service ready!')\"")

if __name__ == "__main__":
    main() 