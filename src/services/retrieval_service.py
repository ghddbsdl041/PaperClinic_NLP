import os
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


class RetrievalService:
    def __init__(self, embedding_model: str = "text-embedding-3-small") -> None:
        self._embeddings = OpenAIEmbeddings(
            model=embedding_model,
            api_key=os.environ["OPENAI_API_KEY"],
        )

    def build_store(self, splits: list[Document]) -> FAISS:
        return FAISS.from_documents(splits, self._embeddings)

    def retrieve(self, store: FAISS, query: str, k: int = 4) -> list[Document]:
        return store.similarity_search(query, k=k)


if __name__ == "__main__":
    # 프로젝트 루트에서, .env에 OPENAI_API_KEY 설정 후:
    # python -m src.services.retrieval_service
    from pathlib import Path

    from dotenv import load_dotenv

    load_dotenv()

    root = Path(__file__).resolve().parents[2]
    sample = root / "sample.pdf"
    if not sample.is_file():
        print(f"샘플 PDF가 없습니다: {sample}")
        raise SystemExit(1)

    from src.services.document_service import DocumentService

    doc_svc = DocumentService()
    docs = doc_svc.load(str(sample))
    splits = doc_svc.split(docs)
    svc = RetrievalService()
    store = svc.build_store(splits)
    q = "이 수업의 목표는 무엇인가?"
    hits = svc.retrieve(store, q, k=2)
    print("질의:", q)
    for i, d in enumerate(hits, 1):
        print(f"\n--- 결과 {i} ---\n{d.page_content[:400]}...")
