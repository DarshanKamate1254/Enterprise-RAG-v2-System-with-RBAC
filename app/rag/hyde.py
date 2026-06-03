"""
RAG v2: HyDE — Hypothetical Document Embeddings.

Instead of embedding the raw user question (which may be short / ambiguous),
HyDE asks the LLM to write a *hypothetical answer*, then embeds that answer.
The embedding of a fluent answer is much closer to real document embeddings
than a short question embedding, which dramatically improves retrieval recall.

Reference: Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance
Labels" (2022). https://arxiv.org/abs/2212.10496
"""

import re
from app.utils.config import get_settings


def generate_hypothetical_document(question: str, llm) -> str:
    """
    Use the LLM to generate a short hypothetical answer to *question*.
    This answer is then used as the retrieval query instead of the raw question.
    """
    hyde_prompt = (
        "Write a short, factual paragraph (3-5 sentences) that would directly "
        f"answer the following question:\n\n{question}\n\n"
        "Paragraph:"
    )
    try:
        response = llm.complete(hyde_prompt)
        hypothesis = str(response).strip()
        # Prepend question so we don't lose keyword signal
        return f"{question}\n\n{hypothesis}"
    except Exception as exc:
        print(f"  [HyDE] Failed, using raw query: {exc}")
        return question


def expand_queries(question: str, llm, n: int = 3) -> list[str]:
    """
    RAG v2: Multi-query expansion.
    Generate *n* paraphrases/sub-questions so the retriever covers more ground.
    Returns the original question plus the expansions.
    """
    prompt = (
        f"Generate {n} different ways to ask the following question. "
        "Each variation should capture a different aspect or phrasing.\n"
        f"Original: {question}\n"
        f"Output exactly {n} lines, one question per line, no numbering."
    )
    try:
        response = llm.complete(prompt)
        lines = [l.strip() for l in str(response).strip().splitlines() if l.strip()]
        expansions = lines[:n]
        return [question] + expansions
    except Exception as exc:
        print(f"  [QueryExpansion] Failed, using original: {exc}")
        return [question]
