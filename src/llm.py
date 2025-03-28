import os
from dotenv import load_dotenv
from google import genai
from google.genai import types # Ensure types is imported

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Define the system message
SYSTEM_MESSAGE = "Anda adalah chatbot AI yang mewakili Yayasan Kitiran. Jawablah pertanyaan berdasarkan dokumen yang disediakan dan selalu gunakan Bahasa Indonesia."

# Modified function signature to accept history
def generate_answer(query, context, history):
    """Generates an answer based on query, optional context, and conversation history using a single prompt string."""
    if not GOOGLE_API_KEY:
        return "Error: GOOGLE_API_KEY not found in environment variables."

    try:
        # Reverted to using genai.Client as originally used
        client = genai.Client(api_key=GOOGLE_API_KEY)

        # --- Construct single prompt string with history and context ---
        prompt_parts = []
        # Add history turns
        for item in history[:-1]: # Exclude the last item (current user query)
            role_label = "User" if item['role'] == 'user' else "Assistant"
            prompt_parts.append(f"{role_label}: {item['content']}")
        # Add context if available
        if context:
            prompt_parts.append(f"Context: {context}")
        # Add the current query
        prompt_parts.append(f"User: {query}") # Use 'User' label for consistency
        # Add the final prompt for the answer
        prompt_parts.append("Assistant:")
        # Join all parts into a single string
        prompt_string = "\n".join(prompt_parts)
        # --- End Prompt Construction ---

        # Create the generation config with the system instruction
        generation_config = types.GenerateContentConfig(
            system_instruction=SYSTEM_MESSAGE
            # Add other config like temperature, max_output_tokens if needed
        )

        # Using the original client.models.generate_content call with the string prompt and config
        response = client.models.generate_content(
            model='gemini-2.0-flash-001', # Using original model name
            contents=prompt_string, # Pass the single formatted string
            config=generation_config # Pass the config object
            )

        # Basic response text extraction (using original logic)
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