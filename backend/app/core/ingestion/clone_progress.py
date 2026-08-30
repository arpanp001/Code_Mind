# backend/app/core/ingestion/clone_progress.py
#
# Tracks git clone progress and stores it for polling.
# GitPython's RemoteProgress fires events during cloning;
# we capture those and make the percentage accessible via a dict.

from git import RemoteProgress
from app.utils.logger import get_logger

logger = get_logger(__name__)

# In-memory store: { project_id: percentage (0-100) }
# Simple dict is fine — this is ephemeral data (only needed during cloning)
_clone_progress: dict[str, int] = {}


class CloneProgressTracker(RemoteProgress):
    """
    GitPython RemoteProgress subclass that captures clone progress.

    GitPython calls update() repeatedly during a git clone operation.
    We convert the raw op_code and cur_count into a 0-100 percentage
    and store it in _clone_progress so the status endpoint can read it.
    """

    def __init__(self, project_id: str):
        super().__init__()
        self.project_id = project_id
        # Initialize at 0 so the endpoint shows 0% immediately
        _clone_progress[project_id] = 0

    def update(
        self,
        op_code: int,
        cur_count: int | str,
        max_count: int | str | None = None,
        message: str = ""
    ):
        """
        Called by GitPython as clone progresses.

        op_code: bit flags for the current operation
                 (COUNTING, COMPRESSING, RECEIVING, RESOLVING)
        cur_count: objects processed so far
        max_count: total objects to process
        message:   human-readable progress message from git
        """
        if max_count and int(max_count) > 0:
            percentage = int((int(cur_count) / int(max_count)) * 100)
            # Clamp to 0-99 — we set 100% only after clone completes
            percentage = max(0, min(99, percentage))
            _clone_progress[self.project_id] = percentage

            # Log every 25% to avoid log spam
            if percentage % 25 == 0:
                logger.debug(
                    f"Clone progress [{self.project_id}]: "
                    f"{percentage}% {message}"
                )

    def complete(self):
        """Call this after clone finishes to set progress to 100%."""
        _clone_progress[self.project_id] = 100
        logger.info(f"✅ Clone complete: {self.project_id}")

    def cleanup(self):
        """Remove progress entry after project is fully processed."""
        _clone_progress.pop(self.project_id, None)


def get_clone_progress(project_id: str) -> int:
    """
    Returns the current clone progress percentage (0-100).
    Returns -1 if no clone is in progress for this project.
    """
    return _clone_progress.get(project_id, -1)