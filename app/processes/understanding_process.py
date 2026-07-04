from app.services.understanding_service import StructuredUnderstanding, UnderstandingService


class UnderstandingProcess:
    """Input-only process. It does not call mem0 or emit events."""

    def __init__(self, understanding_service: UnderstandingService) -> None:
        self._understanding_service = understanding_service

    def run(self, message: str) -> StructuredUnderstanding:
        return self._understanding_service.parse(message)
