import ollama
import asyncio
import json
from typing import Dict, List, Any, Optional
import httpx

# Global ollama service instance
_ollama_service = None

async def get_ollama_response(prompt_data: Dict) -> str:
    """Get response from Ollama using global service instance"""
    global _ollama_service
    
    try:
        # Initialize service if needed
        if _ollama_service is None:
            _ollama_service = OllamaService()
            await _ollama_service.initialize()
        
        # Extract prompt components
        prompt = prompt_data.get('prompt', '')
        prompt_type = prompt_data.get('type', 'general')
        
        # Add type-specific system prompts
        system_prompts = {
            'risk_analysis': """You are a professional crypto risk analyst. 
                              Focus on identifying and quantifying risks, providing specific mitigation strategies.""",
            
            'opportunity_analysis': """You are a professional crypto market analyst.
                                     Focus on identifying high-conviction opportunities backed by data.""",
            
            'strategy_recommendations': """You are a professional portfolio strategist.
                                         Focus on actionable recommendations with clear rationale.""",
            
            'general': """You are a professional crypto market analyst.
                         Provide clear, concise, and actionable insights."""
        }
        
        system_prompt = system_prompts.get(prompt_type, system_prompts['general'])
        
        # Generate response
        response = await _ollama_service.generate_response(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.7
        )
        
        return response
        
    except Exception as e:
        print(f"⚠️  Error getting Ollama response: {e}")
        return "Error: Could not generate AI response. Please check Ollama service."

class OllamaService:
    def __init__(self, model_name: str = "llama2", host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.host = host
        self.client = ollama.Client(host=host)
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize and verify Ollama connection"""
        try:
            # Check if Ollama is running
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.host}/api/tags")
                if response.status_code == 200:
                    models = response.json()
                    available_models = [model["name"] for model in models.get("models", [])]
                    
                    # Check if our preferred model is available
                    if self.model_name not in available_models:
                        print(f"⚠️  Model {self.model_name} not found. Available models: {available_models}")
                        if available_models:
                            self.model_name = available_models[0]
                            print(f"🔄 Using {self.model_name} instead")
                        else:
                            print("❌ No models available. Please pull a model first:")
                            print("   ollama pull llama2")
                            return False
                    
                    self.is_initialized = True
                    print(f"✅ Ollama connected successfully with model: {self.model_name}")
                    return True
                else:
                    print("❌ Ollama server not responding")
                    return False
                    
        except Exception as e:
            print(f"❌ Failed to connect to Ollama: {e}")
            print("Make sure Ollama is running: 'ollama serve'")
            return False

    async def generate_response(self, prompt: str, system_prompt: str = None, temperature: float = 0.7) -> str:
        """Generate response from Ollama model"""
        if not self.is_initialized:
            await self.initialize()
            
        try:
            messages = []
            
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            messages.append({
                "role": "user", 
                "content": prompt
            })
            
            # Use asyncio to run the synchronous client call
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat(
                    model=self.model_name,
                    messages=messages,
                    options={
                        "temperature": temperature,
                        "top_p": 0.9,
                        "top_k": 40,
                    }
                )
            )
            
            return response["message"]["content"]
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I'm having trouble connecting to my AI brain right now. Please try again!"

    async def generate_structured_response(self, prompt: str, system_prompt: str = None, 
                                         response_format: Dict = None) -> Dict[str, Any]:
        """Generate structured response (attempts to parse JSON)"""
        if response_format:
            format_instruction = f"\n\nPlease respond in JSON format matching this structure: {json.dumps(response_format, indent=2)}"
            prompt += format_instruction
            
        response = await self.generate_response(prompt, system_prompt, temperature=0.3)
        
        # Try to parse as JSON
        try:
            # Look for JSON in the response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                # Fallback: return as text response
                return {"response": response, "structured": False}
                
        except json.JSONDecodeError:
            return {"response": response, "structured": False}

    async def get_available_models(self) -> List[str]:
        """Get list of available models"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.host}/api/tags")
                if response.status_code == 200:
                    models = response.json()
                    return [model["name"] for model in models.get("models", [])]
        except Exception as e:
            print(f"Error getting models: {e}")
            return []

    def set_model(self, model_name: str):
        """Change the active model"""
        self.model_name = model_name
        print(f"🔄 Switched to model: {model_name}")

    async def pull_model(self, model_name: str) -> bool:
        """Pull a new model from Ollama registry"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.client.pull(model_name)
            )
            print(f"✅ Successfully pulled model: {model_name}")
            return True
        except Exception as e:
            print(f"❌ Failed to pull model {model_name}: {e}")
            return False