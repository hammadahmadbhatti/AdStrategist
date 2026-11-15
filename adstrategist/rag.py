"""Retrieval-augmented generation layer, built on LangChain components.

    crawled pages -> RecursiveCharacterTextSplitter -> local embeddings (Ollama)
                  -> Chroma vector store (one per run, on disk) -> retriever -> passages with scores

Embeddings always run locally, so page text is never sent to a paid service just to be indexed.
The whole layer is optional: without the packages or the embedding model, steps fall back to
trimming page text and the run still completes.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

from .config import Settings
from .textutil import clean, registrable_domain

CHUNK_SIZE = 400
CHUNK_OVERLAP = 60
COLLECTION = "run"

# Retrieval prompts some embedding models were trained with. Leaving them out costs accuracy.
PREFIXES = {
    "nomic-embed-text": ("search_document: ", "search_query: "),
    "embeddinggemma": ("title: none | text: ", "task: search result | query: "),
    "mxbai-embed-large": ("", "Represent this sentence for searching relevant passages: "),
}


class RagUnavailable(RuntimeError):
    """Retrieval cannot be used on this machine right now. The message says why and how to fix it."""


@dataclass
class Passage:
    text: str
    url: str
    site: str      # "own", a competitor's domain, or "report"
    title: str
    kind: str      # "page" or "report"
    score: float   # cosine similarity to the query, 0 to 1

    def to_dict(self) -> dict:
        return asdict(self)


def _prefixed(base, model: str):
    """Wrap an Embeddings object so documents and queries get the model's own retrieval prefixes."""
    from langchain_core.embeddings import Embeddings

    doc_prefix, query_prefix = next((v for k, v in PREFIXES.items() if model.split(":")[0] == k), ("", ""))
    if not (doc_prefix or query_prefix):
        return base

    class Prefixed(Embeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return base.embed_documents([doc_prefix + t for t in texts])

        def embed_query(self, text: str) -> list[float]:
            return base.embed_query(query_prefix + text)

    return Prefixed()


def rag_status(settings: Settings) -> dict:
    """Whether retrieval can run, in words the web form and CLI can show."""
    if settings.rag == "off":
        return {"ready": False, "detail": "Retrieval is switched off (RAG=off in .env)."}
    try:
        import langchain_chroma  # noqa: F401
        import langchain_ollama  # noqa: F401
        import langchain_text_splitters  # noqa: F401
    except ImportError:
        return {"ready": False, "detail": 'The retrieval packages are not installed. Run: pip install -e ".[rag]"'}
    model = settings.embed_model
    try:
        names = [m.get("name", "") for m in httpx.get(f"{settings.ollama_host}/api/tags", timeout=3.0).json().get("models", [])]
    except (httpx.HTTPError, ValueError):
        return {"ready": False, "detail": "Ollama is not running, so pages cannot be embedded. Start it with: ollama serve"}
    if model in names or f"{model}:latest" in names:
        return {"ready": True, "detail": f"Pages are embedded locally with {model} and stored in a Chroma index."}
    return {"ready": False, "detail": f"The embedding model {model} is not installed. Run: ollama pull {model}"}


def make_embeddings(settings: Settings):
    status = rag_status(settings)
    if not status["ready"]:
        raise RagUnavailable(status["detail"])
    from langchain_ollama import OllamaEmbeddings

    return _prefixed(OllamaEmbeddings(model=settings.embed_model, base_url=settings.ollama_host), settings.embed_model)


class RagIndex:
    """One Chroma collection per run, holding chunks of every page that was read plus the final report."""

    def __init__(self, directory: Path, embeddings, model: str = "") -> None:
        from chromadb.config import Settings as ChromaSettings
        from langchain_chroma import Chroma
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self.model = model
        self.store = Chroma(
            collection_name=COLLECTION, embedding_function=embeddings, persist_directory=str(directory),
            collection_metadata={"hnsw:space": "cosine"},
            client_settings=ChromaSettings(anonymized_telemetry=False, is_persistent=True,
                                           persist_directory=str(directory)),
        )
        # Crawled text has no paragraph breaks left, so split on sentence ends first, then smaller units.
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, keep_separator="end",
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "])
        self.retriever = self.store.as_retriever(search_kwargs={"k": 4})
        self.sources: set[str] = set()

    @classmethod
    def open(cls, directory: Path, settings: Settings, embeddings=None) -> "RagIndex":
        """Create or reopen the index. Raises RagUnavailable with a fix-it message when it cannot."""
        try:
            return cls(directory, embeddings or make_embeddings(settings), settings.embed_model)
        except ImportError as exc:
            raise RagUnavailable('The retrieval packages are not installed. Run: pip install -e ".[rag]"') from exc

    def add_text(self, text: str, *, url: str, site: str, title: str = "", kind: str = "page") -> int:
        from langchain_core.documents import Document

        text = text.strip()
        if len(text) < 40:
            return 0
        chunks = [c.strip() for c in self.splitter.split_text(text) if len(c.strip()) >= 40]
        docs, ids = [], []
        for position, chunk in enumerate(chunks):
            docs.append(Document(page_content=chunk, metadata={"url": url, "site": site, "title": title[:200],
                                                                "kind": kind, "position": position}))
            ids.append(hashlib.sha1(f"{url}#{position}".encode()).hexdigest())  # re-adding a page replaces it
        if docs:
            self.store.add_documents(docs, ids=ids)
            self.sources.add(url)
        return len(docs)

    def add_page(self, page, site: str) -> int:
        """Index one crawled page. `site` is "own" or the competitor's domain."""
        if not page.ok:
            return 0
        lead = clean(f"{page.title}. {page.description}")
        return self.add_text(f"{lead} {page.text}", url=page.final_url or page.url, site=site, title=page.title)

    def search(self, query: str, k: int = 4, *, site: str = "", exclude_site: str = "", kind: str = "") -> list[Passage]:
        """Most similar passages, best first. Filters narrow the search to one site or keep one out."""
        conditions = []
        if site:
            conditions.append({"site": site})
        if exclude_site:
            conditions.append({"site": {"$ne": exclude_site}})
        if kind:
            conditions.append({"kind": kind})
        where = None if not conditions else conditions[0] if len(conditions) == 1 else {"$and": conditions}
        try:
            hits = self.store.similarity_search_with_score(query, k=k, filter=where)
        except Exception:
            return []
        passages = []
        for doc, distance in hits:
            meta = doc.metadata
            passages.append(Passage(doc.page_content, meta.get("url", ""), meta.get("site", ""), meta.get("title", ""),
                                    meta.get("kind", "page"), round(max(0.0, min(1.0, 1.0 - float(distance))), 3)))
        return passages

    def count(self) -> int:
        try:
            return self.store._collection.count()
        except Exception:
            return 0

    def stats(self) -> dict:
        return {"enabled": True, "embedding_model": self.model, "vector_store": "Chroma", "chunks": self.count(),
                "documents": len(self.sources), "chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP}


def site_of(url: str, own_domain: str) -> str:
    domain = registrable_domain(url)
    return "own" if domain == own_domain else domain


def join_passages(passages: list[Passage], limit: int = 2400) -> str:
    """Passages as prompt context: de-duplicated, in reading order per page, within a character budget."""
    seen, kept, used = set(), [], 0
    for passage in passages:
        key = passage.text[:80]
        if key in seen or used + len(passage.text) > limit:
            continue
        seen.add(key)
        kept.append(passage)
        used += len(passage.text)
    return "\n".join(f"- {p.text}" for p in kept)
