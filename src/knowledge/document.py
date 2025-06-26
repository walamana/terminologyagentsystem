import asyncio
import os
import tempfile
from pathlib import Path
from typing import AsyncIterable

from pypdf import PdfReader, PdfWriter

from src.logger import logger
from src.terminology.event import Event
from src.terminology.terminology import DocumentAdded, TextExtracted, TextExtractor, Blackboard
from src.utils import lazy_module


def get_document_converter():
    module = lazy_module("docling.document_converter")
    from tqdm import tqdm
    tqdm(disable=True, total=0)
    return module.DocumentConverter


class Pdf2Text(TextExtractor):

    def extract_text(self, path: str):
        converter = get_document_converter()()
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

        paths = self.split_into_pages(event.path, tempfile.gettempdir())

        logger.info(f"Found {len(paths)} pages in {event.path}")

        tasks = [asyncio.to_thread(self.extract_text, path) for path in paths]

        for task in asyncio.as_completed(tasks):
            path, text = await task
            os.unlink(path)
            yield TextExtracted(text=text)


if __name__ == "__main__":
    blackboard = Blackboard()
    pdf2text = Pdf2Text(blackboard=blackboard)


    async def test():
        counter = 0
        async for event in pdf2text.activate(DocumentAdded(path="./../../data/Handbuch-40820-data_43.pdf")):
            counter += 1
            with open(f"./../../data/Handbuch-40820-data_43-{counter}.txt", "w") as f:
                f.write(event.text)

    asyncio.run(test())