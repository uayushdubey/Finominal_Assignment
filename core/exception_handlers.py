import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from core.exceptions import AppException

logger = logging.getLogger("app.exception_handlers")

def register_exception_handlers(app: FastAPI) -> None:
    """
    Hook exception handlers to the FastAPI app instance.
    """
    @app.exception_handler(AppException)
    async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
        # Standard warning for client/domain errors (4xx), error for server bugs (5xx)
        if exc.status_code >= 500:
            logger.error(
                f"AppException raised: {exc.message} [URL: {request.url.path}]",
                exc_info=True,
                extra={"status_code": exc.status_code, "details": exc.details}
            )
        else:
            logger.warning(
                f"AppException raised: {exc.message} [URL: {request.url.path}]",
                extra={"status_code": exc.status_code, "details": exc.details}
            )
            
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": {
                    "code": exc.__class__.__name__,
                    "message": exc.message,
                    "details": exc.details
                }
            }
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        logger.warning(
            f"Validation error [URL: {request.url.path}]",
            extra={"errors": exc.errors()}
        )
        
        # Format Pydantic errors into a simpler, readable format
        formatted_errors = []
        for error in exc.errors():
            # loc contains location path e.g. ["body", "portfolio", "assets", 0, "weight"]
            # Exclude the source location keyword if desired for readability, or keep it
            field_path = " -> ".join(str(loc) for loc in error.get("loc", []))
            raw_msg = error.get("msg", "")
            
            error_entry = {
                "field": field_path,
                "type": error.get("type", "unknown"),
                "message": raw_msg
            }
            
            # Check if this message was generated from our structured Exception serialized to JSON
            prefix = "Value error, "
            if raw_msg.startswith(prefix):
                json_candidate = raw_msg[len(prefix):]
                if json_candidate.startswith("{") and json_candidate.endswith("}"):
                    try:
                        import json
                        parsed_details = json.loads(json_candidate)
                        if isinstance(parsed_details, dict):
                            error_entry.update(parsed_details)
                            if "error" in parsed_details:
                                error_entry["message"] = parsed_details["error"]
                    except Exception:
                        pass
            
            formatted_errors.append(error_entry)
            
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "error": {
                    "code": "ValidationError",
                    "message": "Input validation failed.",
                    "details": {"errors": formatted_errors}
                }
            }
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.critical(
            f"Unhandled exception occurred [URL: {request.url.path}] | Error: {str(exc)}",
            exc_info=True
        )
        
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": {
                    "code": "InternalServerError",
                    "message": "An unexpected error occurred. Please contact system support.",
                    "details": {}
                }
            }
        )
