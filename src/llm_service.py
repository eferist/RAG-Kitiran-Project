# src/llm_service.py
import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Import prompt templates and constants
from src import prompts

# Load environment variables (consider moving to a central config loader later)
load_dotenv()

class LLMService:
    """
    Encapsulates interactions with the Generative AI model (Google Gemini).
    Handles prompt formatting, API calls, and response parsing for various tasks.
    """
    def __init__(self):
        """
        Initializes the LLMService, loading the necessary API key.
        """
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            # Consider raising a custom exception for better error handling
            print("Error: GOOGLE_API_KEY not found in environment variables.")
            # Or set a flag to prevent API calls
            self.api_key = None # Ensure methods check for this

        # Initialize the client here if it's efficient and thread-safe for your app
        # Or initialize it within each method if needed
        # For now, let's initialize within methods for simplicity, mirroring original code
        # self.client = genai.Client(api_key=self.api_key) if self.api_key else None

    def _get_client(self):
        """Helper to get an initialized client or handle missing API key."""
        if not self.api_key:
            # Raise an error or return None to signal inability to proceed
            raise ValueError("LLM Service cannot operate without GOOGLE_API_KEY.")
        # Consider reusing a client instance if appropriate
        return genai.Client(api_key=self.api_key)

    def generate_answer(self, query: str, context: str | None, history: list[dict]) -> str:
        """
        Generates an answer based on query, optional context, and conversation history.

        Args:
            query: The user's current query.
            context: Retrieved relevant context (e.g., answers from QA pairs), or None.
            history: A list of previous conversation turns [{'role': 'user'/'assistant', 'content': ...}].

        Returns:
            The generated answer string, or an error message.
        """
        try:
            client = self._get_client()

            # Construct single prompt string with history and context
            prompt_parts = []
            # Add history turns (excluding the current user query if it's the last item)
            processed_history = history[:-1] if history and history[-1].get('role') == 'user' else history
            for item in processed_history:
                role_label = "User" if item['role'] == 'user' else "Assistant"
                prompt_parts.append(f"{role_label}: {item['content']}")
            # Add context if available
            if context:
                prompt_parts.append(f"Context: {context}")
            # Add the current query
            prompt_parts.append(f"User: {query}")
            # Add the final prompt for the answer
            prompt_parts.append("Assistant:")
            # Join all parts into a single string
            prompt_string = "\n".join(prompt_parts)

            # Create the generation config with the system instruction
            generation_config = types.GenerateContentConfig(
                system_instruction=prompts.SYSTEM_MESSAGE
                # Add other config like temperature, max_output_tokens if needed
            )

            response = client.models.generate_content(
                model='gemini-2.0-flash-001', # Consider making model configurable
                contents=prompt_string,
                config=generation_config
            )

            # Basic response text extraction
            if hasattr(response, 'text'):
                return response.text
            elif response.candidates and response.candidates[0].content.parts:
                 return response.candidates[0].content.parts[0].text
            else:
                print(f"Unexpected response structure in generate_answer: {response}")
                return "Sorry, received an unexpected response structure."

        except Exception as e:
            print(f"Error during content generation in generate_answer: {e}")
            return f"Sorry, I encountered an error generating the response: {e}"

    def generate_qa_pairs(self, chunk: str, max_pairs: int = 3) -> list[dict]:
        """
        Generates question-answer pairs from a given text chunk using the LLM.

        Args:
            chunk: The text chunk to generate QA pairs from.
            max_pairs: The maximum number of QA pairs to request.

        Returns:
            A list of dictionaries [{'question': ..., 'answer': ...}],
            or an empty list if generation fails or no pairs are found.
        """
        try:
            client = self._get_client()

            # Format the prompt using the template
            prompt = prompts.QA_GENERATION_TEMPLATE.format(chunk=chunk, max_pairs=max_pairs)

            generation_config = types.GenerateContentConfig(
                temperature=0.5,
                response_mime_type="application/json"
            )

            response = client.models.generate_content(
                model='gemini-2.0-flash-001', # Consider making model configurable
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
                    return []
            else:
                print(f"Warning: Unexpected response structure from LLM for QA generation: {response}")
                return []

        except Exception as e:
            print(f"Error during QA pair generation: {e}")
            return []

    def translate_query_to_english(self, query: str) -> str:
        """
        Translates a given query (assumed Indonesian) to English using the LLM.

        Args:
            query: The input query string.

        Returns:
            The translated English query string, or the original query if translation fails.
        """
        try:
            client = self._get_client()

            prompt = prompts.TRANSLATE_TO_ENGLISH_TEMPLATE.format(query=query)

            generation_config = types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=100
            )

            response = client.models.generate_content(
                model='gemini-2.0-flash-001', # Consider making model configurable
                contents=prompt,
                config=generation_config
            )

            # Extract translated text
            if response.candidates and response.candidates[0].content.parts:
                translated_text = response.candidates[0].content.parts[0].text.strip()
                if translated_text:
                     print(f"Translated query to English: '{translated_text}'")
                     return translated_text
                else:
                     print("Warning: LLM returned empty translation.")
                     return query # Return original if translation is empty
            else:
                print(f"Warning: Unexpected response structure from LLM during translation: {response}")
                return query

        except Exception as e:
            print(f"Error during query translation: {e}")
            return query

    def route_query(self, query: str) -> str:
        """
        Classifies the user query based on its relevance to Yayasan Kitiran.

        Args:
            query: The user's input query string.

        Returns:
            "RELATED" if the query is relevant.
            "UNRELATED" if the query is off-topic or classification fails.
        """
        try:
            client = self._get_client()

            prompt = prompts.ROUTING_TEMPLATE.format(query=query)

            generation_config = types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=10
            )

            response = client.models.generate_content(
                model='gemini-2.0-flash-001', # Consider making model configurable
                contents=prompt,
                config=generation_config
            )

            # Extract and validate the classification
            if response.candidates and response.candidates[0].content.parts:
                classification = response.candidates[0].content.parts[0].text.strip().upper()
                if classification == "RELATED":
                    print(f"Query classified as: RELATED")
                    return "RELATED"
                elif classification == "UNRELATED":
                    print(f"Query classified as: UNRELATED")
                    return "UNRELATED"
                else:
                    print(f"Warning: Unexpected classification response from LLM: '{classification}'. Defaulting to UNRELATED.")
                    return "UNRELATED"
            else:
                print(f"Warning: Unexpected response structure from LLM during routing: {response}. Defaulting to UNRELATED.")
                return "UNRELATED"

        except Exception as e:
            print(f"Error during query routing: {e}. Defaulting to UNRELATED.")
            return "UNRELATED"