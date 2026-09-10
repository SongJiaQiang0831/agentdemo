"""FastAPI entry point for the personal AI learning assistant."""

import logging
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.schemas import AskRequest, AskResponse, HealthResponse
from app.services import KnowledgeService, get_knowledge_service

logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).with_name("static")

app = FastAPI(
    title="个人 AI 学习助手",
    version="0.1.0",
    description="提供带来源校验的本地知识库问答 API。",
)

# Serve frontend assets from the same process and origin as the API.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.exception_handler(RequestValidationError)
def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Convert framework validation details into the public error contract."""
    first_error = exc.errors()[0] if exc.errors() else {}
    message = str(first_error.get("ctx", {}).get("error") or first_error.get("msg", "请求参数无效"))
    return JSONResponse(
        status_code=422,
        content={"code": "VALIDATION_ERROR", "message": message},
    )


@app.exception_handler(HTTPException)
def handle_http_error(request: Request, exc: HTTPException) -> JSONResponse:
    """Keep operational HTTP errors consistent with the same error DTO."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"code": "UPSTREAM_ERROR", "message": str(exc.detail)},
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return process health without calling external model services."""
    return HealthResponse(status="ok", service="learning-assistant")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Open the usable chat workspace instead of an API-only landing page."""
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/v1/knowledge/ask", response_model=AskResponse)
def ask_knowledge(
    request: AskRequest,
    service: KnowledgeService = Depends(get_knowledge_service),
) -> AskResponse:
    """Handle one validated RAG question through the application service."""
    try:
        # Key step: the controller delegates business logic to the service layer.
        return service.ask(request.question, request.top_k)
    except Exception as exc:
        logger.exception("Knowledge-base request failed")
        # Do not expose API keys, provider responses, or internal stack traces.
        raise HTTPException(status_code=502, detail="知识库服务暂时不可用") from exc
