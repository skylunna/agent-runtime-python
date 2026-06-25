from fastapi import Request
from fastapi.responses import JSONResponse


class CDSSException(Exception):
    """业务异常基类，所有可预测的错误都应该抛出这个或它的子类"""

    def __init__(self, code: str, message: str, status: int = 400):
        self.code = code
        self.message = message
        self.status = status
        super().__init__(message)

class NotFoundError(CDSSException):
    def __init__(self, message: str = "Not found"):
        super().__init__(code="NOT_FOUND", message=message, status=404)

class ValidationError(CDSSException):
    def __init__(self, message: str = "Validation failed"):
        super().__init__(code="VALIDATION_ERROR", message=message, status=400)

class LLMError(CDSSException):
    def __init__(self, message: str = "LLM call failed"):
        super().__init__(code="LLM_ERROR", message=message, status=502)


async def cdss_exception_handler(request: Request, exc: CDSSException):
    return JSONResponse(
        status_code=exc.status,
        content={"code": exc.code, "message": exc.message},
    )

async def generic_exception_handler(request: Request, exc: Exception):
    import logging
    logging.getLogger(__name__).exception("Unhandled exception")
    return JSONResponse(
        status_code=500,
        content={"code": "INTERNAL_ERROR", "message": "Internal server error"},
    )