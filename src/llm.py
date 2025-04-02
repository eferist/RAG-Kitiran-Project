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

import json # Add json import for parsing

# New function to generate QA pairs from a text chunk
def generate_qa_pairs(chunk: str, max_pairs: int = 3):
    """
    Generates question-answer pairs from a given text chunk using the LLM.

    Args:
        chunk: The text chunk to generate QA pairs from.
        max_pairs: The maximum number of QA pairs to request.

    Returns:
        A list of dictionaries, where each dictionary is {'question': ..., 'answer': ...},
        or an empty list if generation fails or no pairs are found.
    """
    if not GOOGLE_API_KEY:
        print("Error: GOOGLE_API_KEY not found for QA generation.")
        return []

    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)

        # Construct the prompt for QA generation
        prompt = f"""You will generate QA pair from the text chunks that will be used for RAG for AI chatbot that represent Yayasan Kitiran. Yayasan Kitiran focuses on oversight the AI implementation in Indonesia.  Based on the following text chunk, generate up to {max_pairs} relevant question-answer pairs.
The questions should be answerable solely from the provided text.
Format the output strictly as a JSON list of objects, where each object has a 'question' key and an 'answer' key. Example: [{{"question": "...", "answer": "..."}}]. Also the generated QA Pairs must be in english!

Text Chunk:
\"\"\"
{chunk}
\"\"\"

JSON Output:
"""

        # Use a simpler generation config for this task, no system message needed
        generation_config = types.GenerateContentConfig(
            # Adjust temperature or other parameters if needed for QA generation
            temperature=0.5, # Slightly creative but still grounded
            response_mime_type="application/json" # Request JSON output directly if supported
        )

        response = client.models.generate_content(
            model='gemini-2.0-flash-001', # Or choose another suitable model
            contents=prompt,
            config=generation_config
        )

        # Attempt to parse the JSON response
        if response.candidates and response.candidates[0].content.parts:
            response_text = response.candidates[0].content.parts[0].text
            try:
                qa_list = json.loads(response_text)
                # Basic validation
                if isinstance(qa_list, list) and all(isinstance(item, dict) and 'question' in item and 'answer' in item for item in qa_list):
                    return qa_list
                else:
                    print(f"Warning: LLM response was valid JSON but not the expected list of QA dicts: {response_text}")
                    return []
            except json.JSONDecodeError:
                print(f"Warning: Failed to decode JSON response from LLM for QA generation: {response_text}")
                # Optional: Add fallback parsing logic here if needed
                return []
        else:
            print(f"Warning: Unexpected response structure from LLM for QA generation: {response}")
            return []

    except Exception as e:
        print(f"Error during QA pair generation: {e}")
        return []