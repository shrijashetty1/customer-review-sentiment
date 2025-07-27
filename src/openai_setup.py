# Mock OpenAI setup for demo purposes
import os

class MockOpenAIClient:
    """Mock OpenAI client for demo purposes"""
    
    def __init__(self):
        self.api_key = "mock-api-key"
    
    class chat:
        class completions:
            @staticmethod
            def create(model=None, messages=None, **kwargs):
                class MockResponse:
                    def __init__(self):
                        self.choices = [MockChoice()]
                
                class MockChoice:
                    def __init__(self):
                        self.message = MockMessage()
                
                class MockMessage:
                    def __init__(self):
                        if messages and len(messages) > 0:
                            user_content = messages[-1].get('content', '').lower()
                            if any(word in user_content for word in ['positive', 'good', 'great', 'excellent', 'amazing']):
                                self.content = "Positive sentiment - Customer is satisfied"
                            elif any(word in user_content for word in ['negative', 'bad', 'terrible', 'awful', 'hate']):
                                self.content = "Negative sentiment - Customer is dissatisfied"
                            else:
                                self.content = "Neutral sentiment - Balanced feedback"
                        else:
                            self.content = "Mock sentiment analysis result for demo"
                
                return MockResponse()

def get_openai_client():
    return MockOpenAIClient()

client = MockOpenAIClient()

print("🚨 DEMO MODE: Using mock OpenAI responses")
