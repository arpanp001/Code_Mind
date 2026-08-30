from dataclasses import dataclass
from typing      import Optional


# ── System Prompts ─────────────────────────────────────────────────────────────

# backend/app/core/llm/prompts.py
# Update system prompt to enforce better citations:

CODEBASE_SYSTEM_PROMPT = """You are CodeMind, an expert code analyst.

When answering questions about code:
- Always cite the EXACT file: `src/path/to/file.py`
- Always cite line numbers when available: (lines 42-58)
- For functions: mention the exact name: `authenticate()`
- For classes: mention the class: `class AuthService`
- Structure answers as:
  1. Direct answer (1-2 sentences)
  2. Implementation details with citations
  3. Related components if relevant

Citation format: "In `src/auth/login.py` (lines 12-24), the `authenticate()` function..."

Rules:
- Only use information from the provided code context and metadata
- If information is not in the context, say "This is not in the indexed files"
- Never guess file paths or function names
- Use markdown: **bold** for emphasis, `backticks` for code names"""

CODE_EXPLANATION_SYSTEM_PROMPT = """You are CodeMind, an expert code \
explainer. Your job is to explain what a piece of code does in clear, \
accurate terms.

Explanation structure:
1. What it does (one sentence summary)
2. How it works (step by step if complex)
3. Key details (parameters, return values, side effects)
4. Dependencies or related components (if visible in context)

Rules:
- Be precise about technical terms
- Explain non-obvious patterns (decorators, generators, async/await)
- If the code has a bug or unusual pattern, note it
- Keep explanations proportional to code complexity
"""

ARCHITECTURE_SYSTEM_PROMPT = """You are CodeMind, an expert software \
architect. Analyze the provided code and answer architecture questions.

Focus on:
- Design patterns used (MVC, repository, factory, etc.)
- Data flow between components
- Dependency relationships
- Why certain technical choices were likely made
- Potential improvements or concerns
"""


# ── Prompt Templates ───────────────────────────────────────────────────────────

@dataclass
class PromptTemplate:
    """
    A complete prompt package ready to send to Gemini.
    Separating system_prompt from user_message lets us use
    Gemini's multi-turn conversation format correctly.
    """
    system_prompt:  str
    user_message:   str
    context_used:   bool = True


class PromptBuilder:
    """
    Builds structured prompts for different query types.

    Usage:
        builder  = PromptBuilder()
        template = builder.build_chat_prompt(
            question = "Where is login implemented?",
            context  = assembled_context.context_text,
        )
    """

    # backend/app/core/llm/prompts.py
# Update chat prompt to be more project-aware:

    # backend/app/core/llm/prompts.py
# Update build_chat_prompt to accept project metadata:

    def build_chat_prompt(
        self,
        question:      str,
        context:       str,
        project_name:  str          = "the project",
        project_meta:  dict | None  = None,
    ) -> PromptTemplate:
        """
        Builds the main RAG chat prompt with optional project metadata.
        project_meta: { source_url, branch, languages, file_count, chunk_count }
        """
        # Build metadata section
        meta_lines = []
        if project_meta:
            if project_meta.get("source_url"):
                meta_lines.append(f"- Repository URL: {project_meta['source_url']}")
            if project_meta.get("branch"):
                meta_lines.append(f"- Indexed branch: {project_meta['branch']}")
            if project_meta.get("languages"):
                meta_lines.append(f"- Languages: {project_meta['languages']}")
            if project_meta.get("file_count"):
                meta_lines.append(f"- Files indexed: {project_meta['file_count']}")

        meta_block = ""
        if meta_lines:
            meta_block = (
                "\n**Project metadata (use this to answer factual questions):**\n"
                + "\n".join(meta_lines)
                + "\n"
            )

        user_message = f"""I am analyzing the **{project_name}** codebase.
{meta_block}
Here are the most relevant code sections:

{context}

Question: {question}

Answer based on the code and metadata above. Reference files as \
`path/to/file.py` and cite line numbers when visible."""

        return PromptTemplate(
            system_prompt = CODEBASE_SYSTEM_PROMPT,
            user_message  = user_message,
            context_used  = True,
        )

    

    def build_chat_prompt_with_history(
        self,
        question:       str,
        context:        str,
        history:        str        = "",
        project_name:   str        = "the project",
    ) -> PromptTemplate:
        """
        Builds chat prompt with conversation history injected.
        Used when the user has asked previous questions in this session.
        """
        history_section = ""
        if history.strip():
            history_section = f"\n{history}\n"

        user_message = f"""I'm working with {project_name} and have a question.

Here is relevant code from the codebase:

{context}
{history_section}
My question: {question}

Please answer based on the code above. If the question refers to \
something from our previous conversation, use that context too. \
Reference specific files and functions in your answer."""

        return PromptTemplate(
            system_prompt = CODEBASE_SYSTEM_PROMPT,
            user_message  = user_message,
            context_used  = True,
        )

    def build_explanation_prompt(
        self,
        code:       str,
        language:   str,
        file_path:  str = "",
        question:   Optional[str] = None,
    ) -> PromptTemplate:
        """
        Builds a prompt specifically for explaining a code snippet.
        Used by the /explain endpoint.
        """
        location = f" from `{file_path}`" if file_path else ""
        focus    = f"\n\nSpecifically: {question}" if question else ""

        user_message = f"""Please explain this {language} code{location}:

````{language}
{code}
```{focus}"""

        return PromptTemplate(
            system_prompt = CODE_EXPLANATION_SYSTEM_PROMPT,
            user_message  = user_message,
            context_used  = False,
        )

    def build_architecture_prompt(
        self,
        question:   str,
        context:    str,
    ) -> PromptTemplate:
        """
        Builds a prompt for architecture/design questions.
        """
        user_message = f"""Analyze the following code and answer this \
architecture question.

Code context:
{context}

Question: {question}"""

        return PromptTemplate(
            system_prompt = ARCHITECTURE_SYSTEM_PROMPT,
            user_message  = user_message,
            context_used  = True,
        )

    def build_no_context_prompt(self, question: str) -> PromptTemplate:
        """
        Fallback prompt when no relevant context was found.
        Tells Gemini to be honest about the limitation.
        """
        user_message = f"""A developer asked this question about their \
codebase, but no relevant code was found in the indexed files:

Question: {question}

Please acknowledge that no relevant code context was found, and suggest:
1. How they might rephrase the question
2. What they might look for manually
3. General guidance if applicable"""

        return PromptTemplate(
            system_prompt = CODEBASE_SYSTEM_PROMPT,
            user_message  = user_message,
            context_used  = False,
        )


# Module-level singleton
prompt_builder = PromptBuilder()