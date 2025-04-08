# src/prompts.py
"""
Centralized store for LLM prompt templates and system messages.
"""

# System message for the main chatbot interaction
SYSTEM_MESSAGE = "Anda adalah chatbot AI yang mewakili Yayasan Kitiran. Jawablah pertanyaan berdasarkan dokumen yang disediakan dan selalu gunakan Bahasa Indonesia."

# Template for generating Question/Answer pairs from a text chunk
QA_GENERATION_TEMPLATE = """
You will generate QA pair from the text chunks that will be used for RAG for AI chatbot that represent Yayasan Kitiran. Yayasan Kitiran focuses on oversight the AI implementation in Indonesia.  Based on the following text chunk, generate up to {max_pairs} relevant question-answer pairs.
The questions should be answerable solely from the provided text.
Format the output strictly as a JSON list of objects, where each object has a 'question' key and an 'answer' key. Example: [{{"question": "...", "answer": "..."}}]. Also the generated QA Pairs must be in english!

Text Chunk:
\"\"\"
{chunk}
\"\"\"

JSON Output:
"""

# Template for translating Indonesian text to English
TRANSLATE_TO_ENGLISH_TEMPLATE = "Translate the following Indonesian text to English:\n\n{query}\n\nEnglish Translation:"

# Template for classifying user query relevance
ROUTING_TEMPLATE = """
Analyze the following user query. Is it primarily asking about or related to 'Yayasan Kitiran', its activities, AI oversight in Indonesia, or related topics?
Answer ONLY with the word 'RELATED' or 'UNRELATED'.

User Query: "{query}"

Classification:
"""