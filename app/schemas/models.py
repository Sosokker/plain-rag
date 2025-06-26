from pydantic import BaseModel, Field
from typing import Optional


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]  # We'll add source tracking later, but good to have now


class IngestResponse(BaseModel):
    message: str
    filename: str


class ConfigUpdateRequest(BaseModel):
    embedding_model: Optional[str] = Field(
        None, description="Name of the embedding model to use"
    )
    reranker_model: Optional[str] = Field(
        None, description="Name of the reranker model to use"
    )
    llm_model: Optional[str] = Field(None, description="Name of the LLM model to use")
    llm_provider: Optional[str] = Field(
        None, description="Name of the LLM provider to use"
    )