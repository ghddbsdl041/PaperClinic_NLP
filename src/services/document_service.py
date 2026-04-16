from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentService:
    def load(self, file_path: str) -> list[Document]:
        loader = PyPDFLoader(file_path)
        return loader.load()

    def split(
        self,
        documents: list[Document],
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return splitter.split_documents(documents)


if __name__ == "__main__":
    # 프로젝트 루트에서: python -m src.services.document_service
    root = Path(__file__).resolve().parents[2]
    sample = root / "sample.pdf"
    if not sample.is_file():
        print(f"샘플 PDF가 없습니다: {sample}")
        raise SystemExit(1)

    svc = DocumentService()
    docs = svc.load(str(sample))
    print(f"로드된 문서 수: {len(docs)}")
    chunks = svc.split(docs)
    print(f"분할 청크 수: {len(chunks)}")
    if chunks:
        print("첫 청크 앞부분:", chunks[0].page_content[:200].replace("\n", " "), "...")
