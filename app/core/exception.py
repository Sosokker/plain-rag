class DocumentInsertionError(Exception):
    """Exception raised when document insertion to database fails."""


class DocumentExtractionError(Exception):
    """Exception raised when document extraction from PDF fails."""


class ModelNotFoundError(Exception):
    """Exception raised when model is not found."""
