from typing import Any, Dict, Optional

class AppException(Exception):
    """
    Base exception for all application-level errors.
    Allows passing standard message, HTTP status code, and dictionary of metadata.
    """
    def __init__(self, message: str, status_code: int = 500, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}

class DomainException(AppException):
    """
    Raised when business rules or constraints are violated.
    Defaults to 422 Unprocessable Entity.
    """
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=422, details=details)

class BadRequestException(AppException):
    """
    Raised for invalid requests, bad inputs, or payload issues.
    Defaults to 400 Bad Request.
    """
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=400, details=details)

class UnauthorizedException(AppException):
    """
    Raised when client authentication is missing or invalid.
    Defaults to 401 Unauthorized.
    """
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=401, details=details)

class ForbiddenException(AppException):
    """
    Raised when authenticated user is forbidden from performing the action.
    Defaults to 403 Forbidden.
    """
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=403, details=details)

class NotFoundException(AppException):
    """
    Raised when a requested resource does not exist.
    Defaults to 404 Not Found.
    """
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=404, details=details)

class ConflictException(AppException):
    """
    Raised when resource state conflicts with the action.
    Defaults to 409 Conflict.
    """
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=409, details=details)

class InvalidInputException(BadRequestException):
    """
    Raised when API request payloads fail domain validation criteria.
    Defaults to 400 Bad Request.
    """
    pass

class ConstraintViolationException(DomainException):
    """
    Raised when optimization limits (weights, bounds, yields) are breached.
    Defaults to 422 Unprocessable Entity.
    """
    pass

class StrategyNotSupportedException(BadRequestException):
    """
    Raised when the optimization strategy requested is not in the system's registry.
    Defaults to 400 Bad Request.
    """
    pass

