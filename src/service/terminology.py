import asyncio
import uuid
from typing import Optional, Annotated, List, AsyncIterable, Type, Any
from uuid import UUID

from pydantic import BaseModel, Field

from src.service.event import Handler, Event, EventDispatcher


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
    occurrences: Annotated[List[UUID], Field(default_factory=list)]
    definitions: Annotated[List[Definition], Field(default_factory=list)]

    def normalized_or_text(self):
        return self.normalization if self.normalization is not None else self.text


class Blackboard(BaseModel):
    terms: Annotated[List[Term], Field(default_factory=list)]
    sources: Annotated[List[TextSource], Field(default_factory=list)]

    def add_term(self, term: str):
        term = Term(text=term)
        self.terms.append(term)
        return term

    def find_term(self, term_str: str):
        for term in self.terms:
            if term.text == term_str:
                return term

    def add_text_source(self, text: str):
        source = TextSource(id=uuid.uuid4(), text=text)
        self.sources.append(source)
        return source


class KnowledgeSource(Handler):
    blackboard: Blackboard

    class Config:
        arbitrary_types_allowed = True


class DocumentAdded(Event):
    path: str


class TextExtracted(Event):
    text: str


class TermExtracted(Event):
    term: Term


class TermsExtracted(Event):
    terms: List[TermExtracted]


class TermNormalized(Event):
    term: Term


class OccurrenceResolved(Event):
    term: Term
    source: TextSource

class VerifiedDefinitionResolved(Event):
    term: Term
    definition: Definition

class PartialDefinitionGenerated(Event):
    term: Term
    definition: Definition


class CombinedDefinitionGenerated(Event):
    term: Term
    combined_definition: Definition
    relevant_definitions: List[Definition]


class TextExtractor(KnowledgeSource):
    handles: Annotated[List[Type[Event]], Field(default_factory=lambda: [DocumentAdded])]

    async def activate(self, event: DocumentAdded) -> AsyncIterable[Event]:
        yield


class TermExtractor(KnowledgeSource):
    handles: Annotated[List[Type[Event]], Field(default_factory=lambda: [TextExtracted])]

    async def activate(self, event: TextExtracted) -> AsyncIterable[Event]:
        yield


class TermNormalizer(KnowledgeSource):
    handles: Annotated[List[Type[Event]], Field(default_factory=lambda: [TermExtracted])]

    async def activate(self, event: TermExtracted) -> AsyncIterable[Event]:
        yield


class OccurrenceResolver(KnowledgeSource):
    handles: Annotated[List[Type[Event]], Field(default_factory=lambda: [TermExtracted, TermNormalized])]

    async def activate(self, event: Event) -> AsyncIterable[Event]:
        yield


class DefinitionResolver(KnowledgeSource):
    handles: Annotated[List[Type[Event]], Field(default_factory=lambda: [TermExtracted, TermNormalized])]

    async def activate(self, event: Event) -> AsyncIterable[Event]:
        yield


class DefinitionGenerator(KnowledgeSource):
    handles: Annotated[List[Type[Event]], Field(default_factory=lambda: [OccurrenceResolved])]

    async def activate(self, event: OccurrenceResolved) -> AsyncIterable[Event]:
        yield


class DefinitionCombiner(KnowledgeSource):
    handles: Annotated[List[Type[Event]], Field(default_factory=lambda: [PartialDefinitionGenerated])]

    async def activate(self, event: PartialDefinitionGenerated) -> AsyncIterable[Event]:
        yield


class Controller:

    def __init__(self):
        self.blackboard = Blackboard()
        self.knowledge_sources = []
        self.broker = EventDispatcher()

    def register_knowledge_source(self, knowledge_source: Type[KnowledgeSource]):
        knowledge_source.blackboard = self.blackboard
        instance = knowledge_source(blackboard=self.blackboard)
        self.knowledge_sources.append(instance)
        self.broker.register_handler(instance)

    async def emit(self, event: Event):
        async with self.broker.task_group:
            self.broker.emit(event)

    async def analyse_document(self, path: str):
        async with self.broker.task_group:
            self.broker.emit(
                DocumentAdded(path=path)
            )

    async def start(self):
        async with self.broker.task_group:
            self.broker.emit(
                TextExtracted(text="Der Schrankenwärter muss das Gleis sichern.")
            )
