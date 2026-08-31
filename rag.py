import os

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient


# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()


# -----------------------------
# Environment variables
# -----------------------------
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# -----------------------------
# Check environment variables
# -----------------------------
if not QDRANT_URL:
    raise ValueError("QDRANT_URL is missing in .env")

if not QDRANT_API_KEY:
    raise ValueError("QDRANT_API_KEY is missing in .env")

if not QDRANT_COLLECTION:
    raise ValueError("QDRANT_COLLECTION is missing in .env")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing in .env")


# -----------------------------
# Embedding model
# MUST be the same model used
# when vectors were created
# -----------------------------
embedding_model = SentenceTransformer("BAAI/bge-m3")

print("BGE-M3 loaded successfully")


# -----------------------------
# Qdrant connection
# -----------------------------
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

print("Connected to Qdrant successfully")


# -----------------------------
# Groq client
# -----------------------------
groq_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY
)

print("Groq client initialized")


# -----------------------------
# RAG function
# -----------------------------
def ask_rag(question: str):

    # 1. Convert question into vector
    query_vector = embedding_model.encode(
        question,
        normalize_embeddings=True
    ).tolist()


    # 2. Search Qdrant
    search_results = qdrant_client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=5,
        with_payload=True
    ).points


    # 3. Check if anything was retrieved
    if not search_results:
        return None


    # 4. Extract retrieved text and sources
    context_parts = []
    sources = []

    for result in search_results:

        payload = result.payload or {}

        text = payload.get("text", "")

        if text:
            context_parts.append(text)

            sources.append({
                "document": payload.get("source"),
                "page": payload.get("page"),
                "section": payload.get("section"),
                "type": payload.get("type")
            })


    # 5. Check if retrieved results contain usable text
    if not context_parts:
        return None


    # 6. Create context
    context = "\n\n".join(context_parts)


    # 7. Create prompt
    prompt = f"""
You are a BIS document assistant.

Answer the user's question using ONLY the information provided in the context.

Context:

{context}

Question:

{question}

Instructions:

1. Give a direct, clear, and concise answer.

2. Use ONLY information available in the context.

3. Do not add information that is not present in the context.

4. Do not repeat the question.

5. Do not mention "context", "documents", or "retrieval".

6. For simple questions, answer in 2-4 sentences.

7. If the question asks for STEPS, a PROCESS, a PROCEDURE,
   or a SEQUENCE:
   - Present each step on a separate line.
   - Use numbered steps.
   - Start each step with "Step 1:", "Step 2:", etc.
   - Keep each step concise.
   - Do not combine multiple steps into one paragraph.

8. If the question asks for multiple items,
   use bullet points.

9. If the answer contains categories or sub-items,
   use nested bullet points when helpful.

10. Use short paragraphs instead of one large block of text.

11. Preserve important technical terms and names
    exactly as they appear in the context.

12. Do not use unnecessary explanations or repetition.

13. If the answer cannot be found in the provided information,
    say exactly:

"I could not find this information in the provided documents."

Formatting example for a process:

Step 1: Preparation
Brief description of the preparation stage.

Step 2: Implementation
Brief description of the implementation stage.

Step 3: Audit
Brief description of the audit stage.

Step 4: Certification
Brief description of the certification stage.
"""


    # 8. Send request to Groq
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        reasoning_effort="low"
    )


    # 9. Get answer
    answer = response.choices[0].message.content


    # 10. Return response
    return {
        "answer": answer,
        "sources": sources
    }