# Backward-compatibility shim.
# All code should import from app.core.config going forward.
from app.core.config import settings as settings  # noqa: F401
