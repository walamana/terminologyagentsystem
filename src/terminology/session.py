import uuid
from typing import Annotated, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from src.knowledge.document import Pdf2Text
from src.knowledge.extract import CValue
from src.knowledge.openai.definition.combiner import OpenAIDefinitionCombiner
from src.knowledge.openai.definition.generator import OpenAIDefinitionGenerator
from src.knowledge.openai.extract import OpenAIExtractor
from src.knowledge.openai.lemmatize import OpenAILemmatizer
from src.knowledge.resolver import CSVDefinitionResolver
from src.terminology.event import DocumentAdded, TextExtracted
from src.terminology.terminology import Controller, Blackboard


class KnowledgeSourcePolicy(BaseModel):
    use_llm: bool = False
    pass

class Session(BaseModel):
    id: Annotated[UUID, Field(default_factory=uuid.uuid4)]
    policy: KnowledgeSourcePolicy

    def setup_controller_document_processing(self, controller: Controller) -> Controller:
        controller.register_knowledge_source(Pdf2Text)
        return controller

    def setup_controller_term_extraction(self, controller: Controller) -> Controller:
        if self.policy.use_llm:
            controller.register_knowledge_source(OpenAIExtractor)
            controller.register_knowledge_source(OpenAILemmatizer)
        else:
            controller.register_knowledge_source(CValue)
        return controller

    def setup_controller_definition_generation(self, controller: Controller) -> Controller:
        controller.register_knowledge_source(CSVDefinitionResolver)
        if self.policy.use_llm:
            controller.register_knowledge_source(OpenAIDefinitionGenerator)
            controller.register_knowledge_source(OpenAIDefinitionCombiner)
        return controller


    async def process_document(self, file_path: str) -> Blackboard:
        controller = Controller()
        self.setup_controller_document_processing(controller)
        self.setup_controller_term_extraction(controller)
        self.setup_controller_definition_generation(controller)

        await controller.emit(DocumentAdded(path=file_path))

        return controller.blackboard


    async def retrieve_term_definition(self, text: str, context: Optional[str] = None) -> Blackboard:
        controller = Controller()
        self.setup_controller_term_extraction(controller)
        self.setup_controller_definition_generation(controller)

        # TODO: Make proper use of context!!!
        if context is not None:
            controller.blackboard.add_text_source(context)

        await controller.emit(TextExtracted(text=text))

        return controller.blackboard

    async def extract_terminology(self, text: str, context: Optional[str] = None) -> Blackboard:
        controller = Controller()
        self.setup_controller_term_extraction(controller)

        # TODO: Make proper use of context!!!
        if context is not None:
            controller.blackboard.add_text_source(context)

        await controller.emit(TextExtracted(text=text))

        return controller.blackboard

    model_config = {
        "arbitrary_types_allowed": True,
    }


class SessionManager:

    sessions = {}

    @staticmethod
    def setup_controller_llm(controller: Controller):
        controller.register_knowledge_source(OpenAIExtractor)
        # controller.register_knowledge_source(CValue)
        controller.register_knowledge_source(OpenAILemmatizer)
        # TODO: Occurrence Resolver
        # controller.register_knowledge_source(OpenAIDefinitionGenerator)
        # controller.register_knowledge_source(OpenAIDefinitionCombiner)

    @classmethod
    def create_session(cls, policy: KnowledgeSourcePolicy) -> Session:
        session = Session(policy=policy)
        cls.sessions[session.id] = session
        return session

    @classmethod
    def remove_session(cls, session_id: UUID):
        cls.sessions.pop(session_id)
