"""Query reformulation service for multi-turn conversational RAG."""
import json
import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI
from .prompts import REFORMULATION_SYSTEM_PROMPT, build_reformulation_user_prompt
from .llm import DEFAULT_LLM_MODEL

logger = logging.getLogger(__name__)


class QueryReformulator:
    """
    Reformulates context-dependent follow-up questions into standalone search queries
    using recent conversation history while preserving original user intent.
    Powered by DeepSeek (deepseek-chat).
    """

    def __init__(
        self,
        llm_client: Optional[OpenAI] = None,
        model: str = DEFAULT_LLM_MODEL,
        max_history_turns: int = 5,
        openai_client: Optional[OpenAI] = None
    ):
        self.llm_client = llm_client or openai_client
        self.model = model
        self.max_history_turns = max_history_turns

    @property
    def openai_client(self) -> Optional[OpenAI]:
        """Backward-compatible alias for llm_client."""
        return self.llm_client

    @openai_client.setter
    def openai_client(self, client: Optional[OpenAI]):
        self.llm_client = client

    def sanitize_history(self, history: Optional[List[Dict[str, Any]]]) -> List[Dict[str, str]]:
        """
        Validate, sanitize, and limit conversation history.
        Only allows 'user' and 'assistant' roles, caps message length,
        and limits to the most recent max_history_turns (up to 2 * max_history_turns messages).
        """
        if not history or not isinstance(history, list):
            return []

        sanitized = []
        for msg in history:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "")).lower().strip()
            content = str(msg.get("content", "")).strip()

            if role in ("user", "assistant") and content:
                # Cap individual message length to 1000 characters to prevent prompt bloat
                sanitized.append({
                    "role": role,
                    "content": content[:1000]
                })

        # Keep only the last (max_history_turns * 2) messages
        max_messages = max(1, self.max_history_turns * 2)
        return sanitized[-max_messages:]

    def reformulate(
        self,
        question: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Produce a standalone search query.
        If conversation history is empty or reformulation fails, returns the original question safely.
        """
        clean_question = question.strip()
        if not clean_question:
            return ""

        sanitized_history = self.sanitize_history(conversation_history)
        if not sanitized_history:
            logger.debug("No conversation history provided. Using original question as search query.")
            return clean_question

        if self.llm_client is None:
            logger.warning("LLM client not configured for QueryReformulator. Using original question.")
            return clean_question

        user_prompt = build_reformulation_user_prompt(
            question=clean_question,
            conversation_history=sanitized_history
        )

        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": REFORMULATION_SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=150
            )

            raw_content = response.choices[0].message.content or "{}"
            parsed = json.loads(raw_content)
            search_query = parsed.get("search_query", "").strip()

            if search_query:
                logger.info(f"Query reformulated: '{clean_question}' -> '{search_query}'")
                return search_query

            logger.warning("Reformulator returned empty 'search_query' field. Using original question.")
            return clean_question

        except Exception as e:
            logger.warning(
                f"Query reformulation call failed ({type(e).__name__}). Falling back safely to original question."
            )
            return clean_question
