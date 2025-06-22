import uuid
from typing import Annotated, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from src.knowledge.definition.combiner import OpenAIDefinitionCombiner
from src.knowledge.definition.generator import OpenAIDefinitionGenerator
from src.knowledge.definition.resolver import CSVDefinitionResolver
from src.knowledge.document import Pdf2Text
from src.knowledge.extract import OpenAIExtractor, CValue
from src.knowledge.lemmatize import OpenAILemmatizer
from src.terminology.event import DocumentAdded, TextExtracted
from src.terminology.terminology import Controller, Blackboard


class KnowledgeSourcePolicy(BaseModel):
    use_llm: bool = False
    pass

class Session(BaseModel):
    id: Annotated[UUID, Field(default_factory=uuid.uuid4)]
    controller: Controller

    async def process_document(self, file_path: str) -> Blackboard:
        await self.controller.emit(DocumentAdded(path=file_path))

        return self.controller.blackboard


    async def retrieve_term_definition(self, text: str, context: Optional[str] = None) -> Blackboard:
        # TODO: Make proper use of context!!!
        if context is not None:
            self.controller.blackboard.add_text_source(context)

        await self.controller.emit(TextExtracted(text=text))

        return self.controller.blackboard

    model_config = {
        "arbitrary_types_allowed": True,
    }


class SessionManager:

    sessions = {}

    @staticmethod
    def setup_controller_llm(controller: Controller):
        controller.register_knowledge_source(OpenAIExtractor)
        controller.register_knowledge_source(CValue)
        controller.register_knowledge_source(OpenAILemmatizer)
        # TODO: Occurrence Resolver
        controller.register_knowledge_source(OpenAIDefinitionGenerator)
        controller.register_knowledge_source(OpenAIDefinitionCombiner)

    @staticmethod
    def base_controller() -> Controller:
        controller = Controller()
        controller.register_knowledge_source(CSVDefinitionResolver)
        controller.register_knowledge_source(Pdf2Text)
        return controller

    @classmethod
    def create_session(cls, policy: KnowledgeSourcePolicy) -> Session:
        controller = cls.base_controller()

        if policy.use_llm:
            cls.setup_controller_llm(controller)

        session = Session(controller=controller)
        cls.sessions[session.id] = session
        return session

    @classmethod
    def remove_session(cls, session_id: UUID):
        cls.sessions.pop(session_id)
