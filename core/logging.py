import json
import logging
import sys
from typing import Any, Dict
from core.config import settings

class StructuredJSONFormatter(logging.Formatter):
    """
    Custom formatter to output structured logs as JSON.
    Useful for production systems integrating with Datadog, ELK, or CloudWatch.
    """
    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%SZ"),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "filename": record.filename,
            "lineno": record.lineno,
        }
        
        # Include exception trace if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
            
        # Add custom extra context passed via extra={}
        if hasattr(record, "__dict__"):
            for key, val in record.__dict__.items():
                if key not in {
                    "args", "asctime", "created", "exc_info", "exc_text", "filename",
                    "funcName", "levelname", "levelno", "lineno", "module", "msecs",
                    "msg", "name", "pathname", "process", "processName", "relativeCreated",
                    "stack_info", "thread", "threadName"
                }:
                    log_data[key] = val
                    
        return json.dumps(log_data)

class ProfessionalConsoleFormatter(logging.Formatter):
    """
    Clean, color-coded text log formatter for local development environments.
    """
    GREY = "\x1b[38;20m"
    YELLOW = "\x1b[33;20m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    GREEN = "\x1b[32;20m"
    CYAN = "\x1b[36;20m"
    RESET = "\x1b[0m"

    COLORS = {
        logging.DEBUG: CYAN,
        logging.INFO: GREEN,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: BOLD_RED
    }

    def format(self, record: logging.LogRecord) -> str:
        log_color = self.COLORS.get(record.levelno, self.GREY)
        asctime = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        message = record.getMessage()
        exc_str = ""
        
        if record.exc_info:
            exc_str = "\n" + self.formatException(record.exc_info)
            
        log_fmt = f"[{log_color}%(levelname)s{self.RESET}] %(asctime)s - %(name)s - %(message)s (%(filename)s:%(lineno)d)"
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record) + exc_str

def setup_logging() -> None:
    """
    Configure root and subsystem loggers to use structured formatters.
    """
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Reset existing handlers to prevent duplicate outputs
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        
    handler = logging.StreamHandler(sys.stdout)
    
    if settings.ENV.lower() in ("production", "staging"):
        handler.setFormatter(StructuredJSONFormatter())
    else:
        handler.setFormatter(ProfessionalConsoleFormatter())
        
    root_logger.addHandler(handler)
    
    # Direct uvicorn and fastapi logs to our root logger config
    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        l = logging.getLogger(logger_name)
        l.handlers = []
        l.propagate = True
        l.setLevel(log_level)
