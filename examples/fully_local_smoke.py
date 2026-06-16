import json
import sys

import httpx


API_URL = "http://localhost:8000"
PDF_PATH = "tests/data/rome_guide.pdf"
FILENAME = "rome_guide.pdf"
QUESTION = "when was rome founded?"


def parse_events(text):
    events = []
    for block in text.strip().split("\n\n"):
        event = {}
        for line in block.splitlines():
            if line.startswith("event: "):
                event["event"] = line.removeprefix("event: ")
            if line.startswith("data: "):
                event["data"] = line.removeprefix("data: ")
        if event:
            events.append(event)
    return events


def get_existing_document_id(client):
    response = client.get(f"{API_URL}/files", timeout=60)
    response.raise_for_status()
    for document in response.json()["documents"]:
        if document["filename"] == FILENAME:
            return document["document_id"]
    return None


def upload_pdf(client):
    with open(PDF_PATH, "rb") as file:
        response = client.post(
            f"{API_URL}/files/upload",
            params={"chunk_size": 1000},
            files={"file": (FILENAME, file, "application/pdf")},
            timeout=600,
        )
    response.raise_for_status()
    payload = response.json()
    print(payload)
    return payload["document_id"]


def main():
    with httpx.Client() as client:
        document_id = get_existing_document_id(client) or upload_pdf(client)
        print({"document_id": document_id})

        response = client.get(
            f"{API_URL}/files/query",
            params={
                "question": QUESTION,
                "temperature": 0,
                "n_docs": 5,
            },
            timeout=300,
        )
        response.raise_for_status()

    events = parse_events(response.text)
    citations = []
    answer = ""
    for event in events:
        if event.get("event") == "token":
            answer += json.loads(event["data"])["text"]
        if event.get("event") == "citations":
            citations = json.loads(event["data"])["citations"]

    if not any("753 B.C." in citation["preview"] for citation in citations):
        raise AssertionError("Expected citation preview containing 753 B.C.")

    print({"answer": answer.strip(), "citations": citations})


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(exc, file=sys.stderr)
        raise
