#!/usr/bin/env python3
"""
Test script for the Exam Score Predictor API
This script demonstrates how to use the API endpoints
"""

import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"

def test_api():
    """Test all API endpoints"""
    
    print("🚀 Testing Exam Score Predictor API")
    print("=" * 50)
    
    # Test 1: Root endpoint
    print("\n1. Testing root endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 2: Health check
    print("\n2. Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 3: Model info
    print("\n3. Testing model info...")
    try:
        response = requests.get(f"{BASE_URL}/model-info")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 4: Training data
    print("\n4. Testing training data...")
    try:
        response = requests.get(f"{BASE_URL}/training-data")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 5: Predictions with different values
    print("\n5. Testing predictions...")
    test_hours = [2.0, 5.0, 8.0, 12.0]
    
    for hours in test_hours:
        try:
            payload = {"hours_studied": hours}
            response = requests.post(
                f"{BASE_URL}/predict",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            print(f"\nHours: {hours}")
            print(f"Status: {response.status_code}")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
        except Exception as e:
            print(f"Error testing {hours} hours: {e}")
    
    # Test 6: Edge cases
    print("\n6. Testing edge cases...")
    edge_cases = [0.0, 24.0, -1.0, 25.0]
    
    for hours in edge_cases:
        try:
            payload = {"hours_studied": hours}
            response = requests.post(
                f"{BASE_URL}/predict",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            print(f"\nEdge case - Hours: {hours}")
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                print(f"Response: {json.dumps(response.json(), indent=2)}")
            else:
                print(f"Error: {response.text}")
        except Exception as e:
            print(f"Error testing edge case {hours}: {e}")

def test_curl_examples():
    """Print example curl commands"""
    print("\n" + "="*50)
    print("📋 CURL COMMAND EXAMPLES")
    print("="*50)
    
    examples = [
        {
            "description": "Get API information",
            "command": f"curl -X GET '{BASE_URL}/'"
        },
        {
            "description": "Check API health",
            "command": f"curl -X GET '{BASE_URL}/health'"
        },
        {
            "description": "Get model details",
            "command": f"curl -X GET '{BASE_URL}/model-info'"
        },
        {
            "description": "Make a prediction (5 hours studied)",
            "command": f"curl -X POST '{BASE_URL}/predict' -H 'Content-Type: application/json' -d '{{\"hours_studied\": 5.0}}'"
        },
        {
            "description": "Get training data",
            "command": f"curl -X GET '{BASE_URL}/training-data'"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['description']}:")
        print(f"   {example['command']}")

if __name__ == "__main__":
    print("📚 Exam Score Predictor API - Test Suite")
    print("Make sure the API is running on http://localhost:8000")
    print("Run: python api.py or uvicorn api:app --host 0.0.0.0 --port 8000")
    
    choice = input("\nDo you want to run the tests? (y/n): ").lower()
    
    if choice == 'y':
        test_api()
    
    test_curl_examples()
    
    print(f"\n🌐 Visit {BASE_URL}/docs for interactive API documentation!")