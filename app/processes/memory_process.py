from typing import Any


class MemoryProcess:
    """Compatibility guard. AgentCore owns memory retrieval and context decisions."""

    def __init__(self, *_: Any, **__: Any) -> None:
        pass

    def retrieve_context(self, *_: Any, **__: Any) -> None:
        raise RuntimeError("MemoryProcess is deprecated. Use AgentCore.build_memory_context().")

    def record_turn(self, *_: Any, **__: Any) -> None:
        raise RuntimeError("MemoryProcess is deprecated. Use AgentCore.run_chat().")
