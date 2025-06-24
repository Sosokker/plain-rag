from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]  # We'll add source tracking later, but good to have now


class IngestResponse(BaseModel):
    message: str
    filename: str
