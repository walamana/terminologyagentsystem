"""

This acts as the TerminologyResource


"""

import os
import shutil
import tempfile
from typing import Optional

from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from starlette.staticfiles import StaticFiles

from src.terminology.session import SessionManager, KnowledgeSourcePolicy
from src.terminology.terminology import Blackboard

app = FastAPI()

app.mount("/demo", StaticFiles(directory="html", html=True), name="demo")

class TextRequestBody(BaseModel):
    text: str
    context: Optional[str] = None


@app.post("/extractTerminology")
async def process_text(request: TextRequestBody) -> Blackboard:
    session = SessionManager.create_session(KnowledgeSourcePolicy(use_llm=True))
    blackboard = await session.extract_terminology(request.text, context=request.context)
    SessionManager.remove_session(session_id=session.id)
    return blackboard

@app.post("/processText")
async def process_text(request: TextRequestBody) -> Blackboard:
    session = SessionManager.create_session(KnowledgeSourcePolicy(use_llm=True))
    blackboard = await session.retrieve_term_definition(request.text, context=request.context)
    SessionManager.remove_session(session_id=session.id)
    return blackboard

@app.post("/processFile")
async def process_file(file: UploadFile) -> Blackboard:
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    session = SessionManager.create_session(KnowledgeSourcePolicy(use_llm=True))

    blackboard = await session.process_document(file_path)

    SessionManager.remove_session(session_id=session.id)

    os.unlink(file_path)

    return blackboard


@app.get("/simple")
async def process_simple(text: str) -> Blackboard:
    session = SessionManager.create_session(KnowledgeSourcePolicy(use_llm=True))
    blackboard = await session.retrieve_term_definition(text)
    SessionManager.remove_session(session_id=session.id)
    return blackboard