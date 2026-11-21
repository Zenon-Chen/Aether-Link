"""Message Gateway (Box 2) — frozen grade_vN arbitration, dual-axis scoring, and nightly evolution."""

from .gateway import MessageGateway
from .policy import GatingPolicyManager

__all__ = ["MessageGateway", "GatingPolicyManager"]
