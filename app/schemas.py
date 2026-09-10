"""HTTP request and response DTOs for the learning assistant API."""

from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    """Stable health-check response used by clients and deployment platforms."""

    status: str
    service: str


class AskRequest(BaseModel):
    """Validate a knowledge-base question at the HTTP boundary."""

    question: str = Field(min_length=1, max_length=1000, description="知识库问题")
    top_k: int = Field(default=4, ge=1, le=10, description="最多检索的文本片段数")

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        """Trim transport whitespace and reject a question with no real content."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("问题不能为空")
        return normalized


class AskResponse(BaseModel):
    """Public response contract; internal LangChain objects never cross the API."""

    answer: str
    citations: list[str]
    retrieved_sources: list[str]
    citation_valid: bool


class ErrorResponse(BaseModel):
    """Unified error body, similar to a Java REST error DTO."""

    code: str
    message: str
