import asyncio
import os
from pathlib import Path
from typing import AsyncIterable

from pypdf import PdfReader, PdfWriter
from docling.document_converter import  DocumentConverter

from src.service.event import Event
from src.logger import logger
from src.service.terminology import DocumentAdded, TextExtracted, TextExtractor

# The internal lock of tqdm has to be initialized, otherwise docling fails. TODO: why exactly?
# see: https://github.com/tqdm/tqdm/issues/457
from tqdm import tqdm
tqdm(disable=True, total=0)



class Pdf2Text(TextExtractor):

    def extract_text(self, path: str):
        converter = DocumentConverter()
        doc = converter.convert(Path(path)).document
        return path, doc.export_to_markdown()

    def split_into_pages(self, path: str, tmp_path: str):
        reader = PdfReader(open(path, "rb"))
        paths = []
        for i in range(len(reader.pages)):
            output = PdfWriter()
            output.add_page(reader.pages[i])
            out_path = f"{tmp_path}/{i}.pdf"
            paths.append(out_path)
            with open(f"{tmp_path}/{i}.pdf", "wb") as file:
                output.write(file)
        return paths

    async def activate(self, event: DocumentAdded) -> AsyncIterable[Event]:
        paths = self.split_into_pages(event.path, "../../tmp")

        logger.info(f"Found {len(paths)} pages in {event.path}")

        tasks = [asyncio.to_thread(self.extract_text, path) for path in paths]

        for task in asyncio.as_completed(tasks):
            path, text = await task
            os.unlink(path)
            yield TextExtracted(text=text)
