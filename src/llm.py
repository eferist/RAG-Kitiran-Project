import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def generate_answer(query, context):
    """Generates an answer based on query and context (single turn)."""
    if not GOOGLE_API_KEY:
        return "Error: GOOGLE_API_KEY not found in environment variables."

    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)
        prompt = f"""
        Context: {context}
        Question: {query}
        Answer:
        """
        # Using your original model name 'gemini-2.0-flash-001'
        response = client.models.generate_content(
            model='gemini-2.0-flash-001', 
            contents=prompt
            )

        # Basic response text extraction
        if hasattr(response, 'text'):
            return response.text
        elif response.candidates and response.candidates[0].content.parts:
             return response.candidates[0].content.parts[0].text
        else:
            print(f"Unexpected response structure: {response}")
            return "Sorry, received an unexpected response structure."

    except Exception as e:
        print(f"Error during content generation: {e}")
        return f"Sorry, I encountered an error generating the response: {e}"