"""Prompt templates and formatters for RAG and general fallback."""
from typing import List, Dict, Any

GROUNDED_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer ONLY using the provided document context. "
    "If the answer cannot be found in the context, explicitly state that the uploaded documents "
    "do not contain enough information to answer. Do not invent facts or use outside knowledge. "
    "Keep answers concise, accurate, and directly relevant."
)

FALLBACK_SYSTEM_PROMPT = (
    "You are a helpful general-purpose assistant."
)

REFORMULATION_SYSTEM_PROMPT = (
    "You are a query reformulation component for a RAG system.\n"
    "Your job is to rewrite the latest user question into a clear, standalone search query.\n\n"
    "Rules:\n"
    "1. Convert the latest user question into a standalone search query that can be understood without conversation history.\n"
    "2. Use conversation history ONLY to resolve ambiguous pronouns or references (e.g. 'it', 'this', 'that', 'they', 'them', 'how much', 'what about it', 'the previous one').\n"
    "3. Preserve important entities, names, products, policies, dates, numbers, and constraints.\n"
    "4. Do NOT answer the question. Do NOT include explanations.\n"
    "5. Do NOT add information that is not present in the conversation history or question.\n"
    "6. If the question is already clear and standalone, return it with minimal or no changes.\n"
    "7. Output ONLY a valid JSON object in the exact format: {\"search_query\": \"<standalone query>\"}"
)


def build_reformulation_user_prompt(question: str, conversation_history: List[Dict[str, str]]) -> str:
    """
    Format conversation history and current question for the query reformulator.
    """
    formatted_turns = []
    for msg in conversation_history:
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "").strip()
        if content:
            formatted_turns.append(f"{role}: {content}")

    history_text = "\n".join(formatted_turns) if formatted_turns else "No prior conversation."

    return (
        f"CONVERSATION HISTORY:\n"
        f"---------------------\n"
        f"{history_text}\n"
        f"---------------------\n\n"
        f"LATEST USER QUESTION:\n"
        f"{question}\n\n"
        f"Standalone Search Query (JSON):"
    )


def build_grounded_user_prompt(sources: List[Dict[str, Any]], question: str) -> str:
    """
    Build a clearly delimited user prompt combining document context and user question.
    """
    context_blocks = []
    for i, src in enumerate(sources, start=1):
        filename = src.get("filename", "unknown")
        page = src.get("page_number")
        page_str = f"Page {page}" if page is not None else "Page N/A"
        chunk_idx = src.get("chunk_index", 0)
        full_text = src.get("full_text", src.get("snippet", ""))

        block = (
            f"[Source {i}: {filename} | {page_str} | Chunk {chunk_idx}]\n"
            f"{full_text}"
        )
        context_blocks.append(block)

    formatted_context = "\n\n".join(context_blocks)

    return (
        "DOCUMENT CONTEXT:\n"
        "==================================================\n"
        f"{formatted_context}\n"
        "==================================================\n\n"
        "USER QUESTION:\n"
        f"{question}\n\n"
        "Please provide a factual answer based strictly on the document context provided above."
    )
