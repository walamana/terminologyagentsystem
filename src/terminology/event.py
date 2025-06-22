import asyncio
import os
from asyncio import TaskGroup
from collections.abc import AsyncIterable
from typing import Dict, Annotated, Type, List

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from src.logger import logger
from src.terminology.models import Term, TextSource, Definition

load_dotenv()
debug = os.getenv("DEBUG") == "true"



class Event(BaseModel):
    pass

class Handler(BaseModel):
    handles: List[Type[Event]]

    async def activate(self, event: Event) -> AsyncIterable[Event]:
        yield None


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

class EventDispatcher:
    handler: Annotated[Dict[Type[Event], List[Handler]], Field(default_factory=dict)]
    task_group: Annotated[TaskGroup, Field(default_factory=TaskGroup)]
    done_event: Annotated[asyncio.Event, Field(default_factory=asyncio.Event)]
    active_handlers: int = 0

    def __init__(self):
        self.handler = {}
        self.task_group = TaskGroup()
        self.done_event = asyncio.Event()
        self.active_handlers = 0

    def register_handler(self, handler: Handler) -> None:
        for event in handler.handles:
            if event not in self.handler:
                self.handler[event] = []
            self.handler[event].append(handler)


    def emit(self, event: Event):
        if type(event) not in self.handler:
            logger.debug(f"No handler found for {event.__class__.__name__}")
            return

        for handler in self.handler[type(event)]:
            async def handle_event(h: Handler):
                self.done_event.clear()
                self.active_handlers += 1
                logger.debug(f"Event {event.__class__.__name__} calls handler {h.__class__.__name__}")
                async for x in h.activate(event):
                    if x is not None:
                        self.emit(x)
                self.active_handlers -= 1

                if self.active_handlers == 0:
                    self.done_event.set()

            self.task_group.create_task(handle_event(handler))
