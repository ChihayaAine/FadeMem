from typing import Optional
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

class LLMInterface:
    def __init__(self, model_name: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        """
        Initialize the LLM Interface with optional OpenAI API support.
        
        Args:
            model_name (str, optional): Name of the model to use. Defaults to 'gpt-3.5-turbo'.
            api_key (str, optional): API key for OpenAI. If not provided, falls back to mock behavior.
        """
        self.model_name = model_name
        self.client = None
        if OpenAI and api_key:
            self.client = OpenAI(api_key=api_key)
        elif not OpenAI:
            print("Warning: 'openai' library not found. Install it with 'pip install openai' for real LLM calls. Using mock responses.")
        else:
            print("Warning: No API key provided. Using mock responses for LLM calls.")
        
    def generate_response(self, query: str, context: str = "") -> str:
        """
        Generate a response based on a query and optional context from memories.
        Uses OpenAI API if available and configured, otherwise falls back to mock implementation.
        
        Args:
            query (str): The user's query or prompt.
            context (str, optional): Additional context from retrieved memories. Defaults to empty string.
            
        Returns:
            str: Generated response from the LLM.
        """
        if self.client:
            try:
                messages = []
                if context and context != "No relevant memories found.":
                    messages.append({"role": "system", "content": f"You are an assistant with access to the following memories:\n{context}\nUse this information to answer the user's query."})
                messages.append({"role": "user", "content": query})
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=500
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Error calling OpenAI API: {e}. Falling back to mock response.")
        
        # Mock response generation if API call fails or is not configured
        if context and context != "No relevant memories found.":
            return f"Based on the context:\n{context}\n\nMy response to '{query}' is: This is a generated answer incorporating the provided memories."
        else:
            return f"My response to '{query}' is: This is a generated answer without specific memory context."
        
    def set_api_key(self, api_key: str) -> None:
        """
        Set or update the API key for accessing the LLM service.
        
        Args:
            api_key (str): API key for the LLM service.
        """
        if OpenAI:
            self.client = OpenAI(api_key=api_key)
        else:
            print("Warning: 'openai' library not installed. Install it to use real LLM calls.") 