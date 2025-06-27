import uuid
from typing import Any, Optional, Annotated, List
from uuid import UUID

from pydantic import BaseModel, Field


class TextSource(BaseModel):
    id: UUID = Field(UUID)
    text: str


class Definition(BaseModel):
    text: str
    verified: bool
    partial: bool
    source: Optional[TextSource | Any] = None

    def is_combined(self) -> bool:
        return not self.verified and not self.partial

    def is_partial(self) -> bool:
        return not self.verified and self.partial

class Term(BaseModel):
    id: Annotated[UUID, Field(default_factory=uuid.uuid4)]
    text: str
    normalization: Optional[str] = None
    variations: Annotated[List[str], Field(default_factory=list)]
    occurrences: Annotated[List[UUID], Field(default_factory=list)]
    definitions: Annotated[List[Definition], Field(default_factory=list)]

    def normalized_or_text(self):
        return self.normalization if self.normalization is not None else self.text
