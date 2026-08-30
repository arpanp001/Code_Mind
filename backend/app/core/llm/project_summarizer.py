# backend/app/core/llm/project_summarizer.py
# COMPLETE REPLACEMENT

from app.core.llm.gemini  import gemini_client
from app.core.llm.prompts import PromptTemplate
from app.utils.logger     import get_logger

logger = get_logger(__name__)


def generate_project_summary(
    project_id:   str,
    project_name: str,
    file_count:   int,
    chunk_count:  int,
    top_files:    list[str],
    languages:    list[str],
    source_url:   str = "",
    branch:       str = "main",
) -> str:
    """
    Generates a rich project overview including:
    - What the project does
    - Key technologies and dependencies
    - Main entry points
    - Folder structure summary
    - Architecture patterns detected
    """
    files_str     = '\n'.join(f'  {f}' for f in top_files[:15])
    languages_str = ', '.join(languages[:6]) if languages else 'unknown'

    # Detect architecture from file paths
    arch_hints = []
    files_lower = [f.lower() for f in top_files]

    if any('router' in f or 'route' in f for f in files_lower):
        arch_hints.append('router-based routing')
    if any('middleware' in f for f in files_lower):
        arch_hints.append('middleware pattern')
    if any('model' in f or 'schema' in f for f in files_lower):
        arch_hints.append('data models/schemas')
    if any('test' in f or 'spec' in f for f in files_lower):
        arch_hints.append('test suite included')
    if any('docker' in f or 'compose' in f for f in files_lower):
        arch_hints.append('Docker containerized')
    if any('config' in f or 'settings' in f for f in files_lower):
        arch_hints.append('configuration management')

    arch_str = ', '.join(arch_hints) if arch_hints else 'standard structure'

    url_line = f"Repository: {source_url} (branch: {branch})" if source_url else ""

    prompt = PromptTemplate(
        system_prompt = (
            "You are CodeMind, an expert code analyst. "
            "Generate a rich, specific project overview in 4-6 sentences. "
            "Include: what it does, main technologies, architecture patterns, "
            "key entry points, and notable design choices. "
            "Be specific — avoid generic descriptions. "
            "Format: 2-3 sentences of what + 1-2 sentences of how + 1 sentence "
            "of notable patterns. Use **bold** for key terms."
        ),
        user_message = (
            f"Project: {project_name}\n"
            f"{url_line}\n"
            f"Languages: {languages_str}\n"
            f"Files indexed: {file_count} ({chunk_count} chunks)\n"
            f"Architecture hints: {arch_str}\n"
            f"Key files sampled:\n{files_str}\n\n"
            f"Generate a specific, informative project overview that would "
            f"help a developer understand this codebase in 30 seconds. "
            f"Include entry points, key patterns, and what makes this project notable."
        ),
    )

    try:
        response = gemini_client.generate(prompt)
        if response.success and response.text:
            return response.text
    except Exception as e:
        logger.warning(f"Rich summary generation failed: {e}")

    # Fallback — construct a reasonable summary from available data
    arch_desc = f" Key patterns: {arch_str}." if arch_hints else ""
    url_desc  = f" Source: [{source_url}]({source_url}) on `{branch}` branch." if source_url else ""
    return (
        f"**{project_name}** is a {languages_str} project with "
        f"{file_count} indexed files.{arch_desc}{url_desc} "
        f"Ask any question to explore the codebase."
    )