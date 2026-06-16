import io
import json
from uuid import uuid4

from fastapi import HTTPException
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from starlette.concurrency import run_in_threadpool

from .. import settings
from .llms import get_chat_model


def partition_pdf_file(file):
    from unstructured.partition.pdf import partition_pdf

    return partition_pdf(file=file)


class FilesService:
    @staticmethod
    def _message_text(message):
        text = getattr(message, "text", None)
        if callable(text):
            return text()
        if text is not None:
            return text
        content = getattr(message, "content", message)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                block.get("text", "") for block in content if isinstance(block, dict)
            )
        return str(content)

    @staticmethod
    def _citation(doc, score=None):
        metadata = doc.metadata
        return {
            "filename": metadata.get("filename"),
            "page": metadata.get("page"),
            "chunk_index": metadata.get("chunk_index"),
            "score": score,
            "preview": doc.page_content[: settings.CITATION_PREVIEW_CHARS].strip(),
        }

    @staticmethod
    def _event(event, payload):
        return f"event: {event}\ndata: {json.dumps(payload)}\n\n"

    @staticmethod
    async def query(question, temperature, n_docs, vectorstore):
        docs_with_scores = await run_in_threadpool(
            vectorstore.similarity_search_with_score,
            question,
            n_docs,
        )
        context = "\n\n".join(
            f"Source {index + 1}: {doc.page_content}"
            for index, (doc, _score) in enumerate(docs_with_scores)
        )
        citations = [
            FilesService._citation(doc, score)
            for doc, score in docs_with_scores
        ]

        messages = [
            (
                "system",
                "Answer using only the provided context. If the context does not contain the answer, say you do not know. Be concise.",
            ),
            ("human", f"Context:\n{context}\n\nQuestion: {question}"),
        ]
        llm = get_chat_model(temperature)

        def stream():
            for chunk in llm.stream(messages):
                text = FilesService._message_text(chunk)
                if text:
                    yield FilesService._event("token", {"text": text})
            yield FilesService._event("citations", {"citations": citations})

        return stream()

    @staticmethod
    async def upload(file, chunk_size, vectorstore):
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")

        data = await FilesService._read_upload(file)
        if not data:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        document_id = str(uuid4())
        elements = await run_in_threadpool(partition_pdf_file, io.BytesIO(data))
        source_docs = []

        for element in elements:
            text = getattr(element, "text", "") or str(element)
            text = text.strip()
            if not text:
                continue
            metadata = getattr(element, "metadata", None)
            doc_metadata = {
                "document_id": document_id,
                "filename": file.filename,
            }
            page = getattr(metadata, "page_number", None)
            if page is not None:
                doc_metadata["page"] = page
            source_docs.append(
                Document(
                    page_content=text,
                    metadata=doc_metadata,
                )
            )

        if not source_docs:
            raise HTTPException(status_code=400, detail="No text could be extracted.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            add_start_index=True,
        )

        docs = text_splitter.split_documents(source_docs)
        if len(docs) > settings.MAX_CHUNKS_PER_UPLOAD:
            raise HTTPException(
                status_code=413,
                detail=f"Document produced {len(docs)} chunks, above the limit of {settings.MAX_CHUNKS_PER_UPLOAD}.",
            )

        for chunk_index, doc in enumerate(docs):
            doc.metadata["chunk_index"] = chunk_index

        for index in range(0, len(docs), settings.VECTORSTORE_ADD_BATCH_SIZE):
            await run_in_threadpool(
                vectorstore.add_documents,
                docs[index : index + settings.VECTORSTORE_ADD_BATCH_SIZE],
            )
        return {
            "document_id": document_id,
            "filename": file.filename,
            "chunks_added": len(docs),
        }

    @staticmethod
    async def _read_upload(file):
        data = bytearray()
        while True:
            chunk = await file.read(settings.UPLOAD_READ_CHUNK_BYTES)
            if not chunk:
                break
            data.extend(chunk)
            if len(data) > settings.MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"Uploaded file exceeds {settings.MAX_UPLOAD_BYTES} bytes.",
                )
        return bytes(data)

    @staticmethod
    async def list(vectorstore, limit, offset):
        documents = await run_in_threadpool(vectorstore.list_documents, limit, offset)
        return {
            "documents": documents,
            "limit": limit,
            "offset": offset,
            "count": len(documents),
        }

    @staticmethod
    async def delete(document_id, vectorstore):
        deleted = await run_in_threadpool(vectorstore.delete_document, document_id)
        if deleted == 0:
            raise HTTPException(status_code=404, detail="Document not found.")
        return {"document_id": document_id, "chunks_deleted": deleted}
