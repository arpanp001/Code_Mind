# backend/app/core/rag/query_expander.py
#
# Generates multiple query variations from a single user question.
# This is called "query expansion" or "multi-query retrieval".
#
# Why it works:
# A single query embedding captures ONE point in 384-dimensional space.
# Different phrasings of the same question capture DIFFERENT points.
# The union of search results from multiple points has much higher recall.
#
# Example:
#   Input:   "where is authentication handled?"
#   Output:  [
#     "where is authentication handled?",          (original)
#     "user login and credential verification",    (semantic variation)
#     "auth middleware security function",         (technical variation)
#     "sign in implementation code location",      (intent variation)
#   ]
#
# We use two strategies:
#   1. Rule-based expansion  (fast, no API call, always available)
#   2. Gemini-based expansion (better quality, requires API, Phase 11+)
# For Phase 10 we implement rule-based only.

import re
from dataclasses import dataclass, field
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ── Query Templates ────────────────────────────────────────────────────────────
#
# These templates rephrase a query from different angles.
# Each slot {query} gets replaced with the cleaned query text.

TECHNICAL_TEMPLATES = [
    "{query}",                                    # Original — always included
    "{query} function implementation",            # Ask for implementation
    "{query} code location file",                 # Ask for location
    "how does {query} work",                      # Ask for explanation
    "{query} method class definition",            # Ask for definition
]

# Domain-specific expansions for common code concepts
DOMAIN_EXPANSIONS = {
    # Auth-related
    "login":          ["authentication", "signin", "credentials", "jwt", "session"],
    "auth":           ["authentication", "authorization", "login", "permission"],
    "authenticate":   ["login", "verify credentials", "jwt token", "session"],
    "password":       ["hashing", "bcrypt", "credentials", "security"],

    # Data-related
    "database":       ["db connection", "orm", "sql", "repository", "model"],
    "query":          ["sql select", "database query", "orm filter", "fetch"],
    "model":          ["schema", "entity", "database table", "orm model"],

    # API-related
    "endpoint":       ["route", "api handler", "controller", "view"],
    "request":        ["http request", "api call", "route handler"],
    "response":       ["http response", "return json", "api response"],

    # Error handling
    "error":          ["exception", "try catch", "error handling", "raise"],
    "exception":      ["error handling", "try except", "catch block"],

    # Architecture
    "middleware":     ["interceptor", "filter", "decorator", "handler"],
    "service":        ["business logic", "service layer", "module"],
    "repository":     ["data access", "dao", "database layer"],
    "config":         ["configuration", "settings", "environment variables"],
    "test":           ["unit test", "pytest", "jest", "assert", "spec"],
}


@dataclass
class ExpandedQuery:
    """
    Result of query expansion.
    Contains the original query and all generated variations.
    """
    original:   str
    variations: list[str]    = field(default_factory=list)
    strategy:   str          = "rule_based"

    @property
    def all_queries(self) -> list[str]:
        """Returns original + unique variations, original first."""
        seen   = {self.original}
        result = [self.original]
        for v in self.variations:
            if v and v not in seen and v.strip():
                seen.add(v)
                result.append(v)
        return result


class QueryExpander:
    """
    Generates query variations using rule-based heuristics.

    This is intentionally simple and fast — no API calls, no ML model.
    It covers the most common cases:
      1. Template variations (rephrase around the same keywords)
      2. Domain synonym expansion (replace domain terms with synonyms)
      3. Code-specific reformulation (add "function", "class", etc.)

    Phase 11 will add Gemini-based expansion for higher quality.
    """

    def __init__(self, max_variations: int = 3):
        """
        Args:
            max_variations: Max number of additional queries beyond original.
                            Total queries = original + max_variations.
                            Keep this ≤ 4 to avoid ChromaDB overload.
        """
        self.max_variations = max_variations

    def expand(self, query: str) -> ExpandedQuery:
        """
        Expands a query into multiple variations.

        Steps:
        1. Clean and normalize the query
        2. Extract key concepts (nouns, verbs)
        3. Apply domain synonym expansion
        4. Generate template-based variations
        5. Deduplicate and return top N

        Args:
            query: The user's original question

        Returns:
            ExpandedQuery with original + variations
        """
        query   = query.strip()
        result  = ExpandedQuery(original=query)

        if not query:
            return result

        variations = []

        # ── Strategy 1: Domain expansion ─────────────────────────────────
        # Find domain terms in the query and add their synonyms
        domain_vars = self._domain_expand(query)
        variations.extend(domain_vars)

        # ── Strategy 2: Template variations ──────────────────────────────
        # Apply templates that rephrase the question
        template_vars = self._template_expand(query)
        variations.extend(template_vars)

        # ── Strategy 3: Code-specific reformulation ───────────────────────
        # Add code-specific context words to improve code retrieval
        code_vars = self._code_reformulate(query)
        variations.extend(code_vars)

        # ── Deduplicate and filter ────────────────────────────────────────
        seen   = {query.lower()}
        unique = []
        for v in variations:
            v_clean = v.strip()
            if v_clean and v_clean.lower() not in seen:
                seen.add(v_clean.lower())
                unique.append(v_clean)

        result.variations = unique[:self.max_variations]

        logger.debug(
            f"Query expanded: '{query[:40]}' "
            f"→ {len(result.all_queries)} queries"
        )
        return result

    def _domain_expand(self, query: str) -> list[str]:
        """
        Replaces domain terms with their synonyms.
        Example: "login" → "authentication", "credential verification"
        """
        variations = []
        query_lower = query.lower()

        for term, synonyms in DOMAIN_EXPANSIONS.items():
            if term in query_lower:
                for synonym in synonyms[:2]:  # Max 2 synonyms per term
                    # Replace the term with its synonym in the query
                    new_query = re.sub(
                        rf'\b{re.escape(term)}\b',
                        synonym,
                        query_lower,
                        flags=re.IGNORECASE
                    )
                    if new_query != query_lower:
                        variations.append(new_query)

        return variations[:2]  # Max 2 from domain expansion

    def _template_expand(self, query: str) -> list[str]:
        """
        Applies rephrase templates to the query.
        Skips the first template (original) since it's already included.
        """
        # Extract core concept (remove question words)
        core = re.sub(
            r'^(where|what|how|which|who|when|why|is|are|does|do)\s+',
            '',
            query.lower()
        ).strip()

        if not core:
            return []

        # Apply non-trivial templates only
        variations = []
        for template in TECHNICAL_TEMPLATES[1:3]:  # Skip original
            variation = template.format(query=core)
            if variation != query:
                variations.append(variation)

        return variations

    def _code_reformulate(self, query: str) -> list[str]:
        """
        Adds code-specific context to the query.
        Helps when users ask in natural language but the code
        uses technical terms.
        """
        query_lower = query.lower()
        variations  = []

        # If query asks "where" → reformulate as implementation search
        if query_lower.startswith(("where", "which file", "what file")):
            core = re.sub(r'^where\s+(is|are|does)\s+', '', query_lower)
            variations.append(f"implement {core}")

        # If query asks "how" → reformulate as function search
        if query_lower.startswith(("how", "explain")):
            core = re.sub(r'^(how|explain)\s+(does|is|the)?\s*', '', query_lower)
            variations.append(f"function {core} implementation")

        # If query contains a function-like word → add "def" for Python
        func_match = re.search(r'\b([a-z_][a-z0-9_]*)\s*\(', query_lower)
        if func_match:
            func_name = func_match.group(1)
            variations.append(f"def {func_name}")

        return variations[:1]  # Max 1 from code reformulation