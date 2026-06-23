#indexing.py
# ───────────────────────────────────────────────────────────────────────────
# Chunking & validation pipeline
# ───────────────────────────────────────────────────────────────────────────
# utils/vector_store.py
from tqdm import tqdm
import sys
import numpy as np
import json
import time
import logging
import argparse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pathlib import Path
from rich.console import Console
import logfire
# ── LangChain core / FAISS ──────────────────────────────────────────────────
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings
# ───────────────────────────────────────────────────────────────────────────
#  Bootstrap
# ───────────────────────────────────────────────────────────────────────────
console = Console()
logging.basicConfig(level=logging.WARNING)
#logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
from utils.models import PipelineConfig, ValidatedChunk, ChunkMetadata
from utils.config import INPUT_DIR, VECTOR_DB_DIR, MISTRAL_API_KEY, SUPPORTED_EXTENSIONS

CORPUS_DIR      = Path(INPUT_DIR)     # PDFs go here
# Paths for persisted artifacts
INDEX_DIR        = Path(VECTOR_DB_DIR)
INDEX_FILE       = INDEX_DIR / "index.faiss"    # raw FAISS binary
CHUNKS_FILE      = INDEX_DIR / "chunks.json"    # validated chunk metadata
CORPUS_HASH_FILE = INDEX_DIR / "corpus.hash"    # SHA-256 of corpus files

# ───────────────────────────────────────────────────────────────────────────
# 1. Chunking & validation pipeline
# ───────────────────────────────────────────────────────────────────────────

@logfire.instrument("chunk_and_validate")
def chunk_and_validate(
    docs: list[Document],
    config: PipelineConfig,
) -> tuple[list[ValidatedChunk], list[Document]]:
    """
    Split docs → validate each chunk with Pydantic → return:
      - valid ValidatedChunk objects  (for reporting)
      - LangChain Documents           (for FAISS indexing)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    raw_chunks = splitter.split_documents(docs)
 
    valid_chunks:    list[ValidatedChunk] = []
    langchain_docs:  list[Document]       = []
    invalid_count = 0
 
    for idx, chunk in enumerate(raw_chunks):
        try:
            meta = ChunkMetadata(
                source    = str(chunk.metadata.get("source", "unknown")),
                page      = int(chunk.metadata.get("page", 0)),
                thread_id = str(chunk.metadata.get("thread_id", "")),
                topic     = str(chunk.metadata.get("topic", "general")),
                chunk_idx = idx,
            )
            validated = ValidatedChunk(
                content  = chunk.page_content,
                metadata = meta,
            )
            valid_chunks.append(validated)
            # Rebuild LangChain Document with cleaned content + validated meta
            langchain_docs.append(Document(
                page_content = validated.content,
                metadata     = validated.metadata.model_dump(),
            ))
        except Exception as exc:
            invalid_count += 1
            logfire.warn("Chunk validation failed", idx=idx, reason=str(exc))
 
    logfire.info(
        "Chunking complete",
        total=len(raw_chunks),
        valid=len(valid_chunks),
        invalid=invalid_count,
    )
    return valid_chunks, langchain_docs


# ───────────────────────────────────────────────────────────────────────────
# 2.FAISS index — build, save, load
# ───────────────────────────────────────────────────────────────────────────

def _corpus_hash(corpus_dir: Path) -> str:
    """
    Stable SHA-256 fingerprint of ALL supported files in corpus_dir.
 
    Cover every extension in SUPPORTED_EXTENSIONS so adding a .csv or .xlsx to the corpus
    correctly invalidates the cached index, not just new PDFs.
 
    Hash is computed over each file's (relative_path + size + mtime),
    sorted for determinism — fast and robust without reading file bytes.
    """
    import hashlib
    h = hashlib.sha256()
    all_files = sorted(
        f for f in corpus_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not all_files:
        logfire.warn("_corpus_hash: no supported files found", path=str(corpus_dir))
    for file_path in all_files:
        stat = file_path.stat()
        # Include relative path so moving a file also invalidates the cache
        entry = f"{file_path.relative_to(corpus_dir)}:{stat.st_size}:{stat.st_mtime}"
        h.update(entry.encode())
    return h.hexdigest()

def _save_index(
    index:        FAISS,
    valid_chunks: list[ValidatedChunk],
    corpus_dir:   Path,
) -> None:
    """Persist FAISS index + chunk metadata + corpus hash to INDEX_DIR."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
 
    # 1. FAISS binary (index.faiss + index.pkl via LangChain helper)
    index.save_local(str(INDEX_DIR))
    logfire.info("FAISS index saved", path=str(INDEX_DIR))
 
    # 2. Validated chunks as JSON (preserves all Pydantic metadata)
    chunks_data = [c.model_dump() for c in valid_chunks]
    CHUNKS_FILE.write_text(
        json.dumps(chunks_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logfire.info("Chunks saved", n=len(chunks_data), path=str(CHUNKS_FILE))
 
    # 3. Corpus fingerprint — lets us detect stale cache on next run
    CORPUS_HASH_FILE.write_text(_corpus_hash(corpus_dir))
    logfire.info("Corpus hash saved")

def _load_index(
    embeddings: MistralAIEmbeddings,
) -> tuple[FAISS, list[ValidatedChunk]] | None:
    """
    Try to load a previously saved FAISS index + chunks from INDEX_DIR.
    Returns (index, chunks) on success, or None on any failure.
 
    Failure cases — each error raises a distinct console warning:
      1. index.faiss missing → index was never built or was deleted
      2. chunks.json missing → chunks file missing independently
      3. FAISS.load_local fails → corrupted binary or schema change
      4. chunks.json parse fails → Pydantic schema changed between runs
      5. Vector/chunk count mismatch → index and chunks are out of sync
    """
    # ── 1. Check files exist before attempting load ────────────────────────
    missing = []
    if not INDEX_FILE.exists():
        missing.append(str(INDEX_FILE))
    if not CHUNKS_FILE.exists():
        missing.append(str(CHUNKS_FILE))
 
    if missing:
        for path in missing:
            logfire.warn(
                "Cache file missing — index must be built from scratch",
                missing_file=path,
            )
        console.print(
            "  [yellow]⚠ Cache miss:[/yellow] "
            + ", ".join(missing)
            + " not found — will build index from scratch."
        )
        return None
 
    # ── 2. Load FAISS binary ──────────────────────────────────────────────
    try:
        index = FAISS.load_local(
            str(INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    except Exception as exc:
        logfire.warn(
            "FAISS index load failed — corrupted binary or schema change",
            reason=str(exc),
            index_path=str(INDEX_FILE),
        )
        console.print(
            f"  [yellow]⚠ Index load failed:[/yellow] {exc}\n"
            "  → Rebuilding index from scratch."
        )
        return None
 
    # ── 3. Load chunks JSON ───────────────────────────────────────────────
    try:
        raw    = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
        chunks = [ValidatedChunk(**c) for c in raw]
    except Exception as exc:
        logfire.warn(
            "Chunks JSON load failed — Pydantic schema may have changed",
            reason=str(exc),
            chunks_path=str(CHUNKS_FILE),
        )
        console.print(
            f"  [yellow]⚠ Chunks file invalid:[/yellow] {exc}\n"
            "  → Rebuilding index from scratch."
        )
        return None
 
    # ── 4. Sanity-check vector / chunk count ─────────────────────────────
    n_vectors = index.index.ntotal
    n_chunks  = len(chunks)
    if n_vectors != n_chunks:
        logfire.warn(
            "Index/chunk count mismatch — cache is corrupt",
            vectors=n_vectors,
            chunks=n_chunks,
        )
        console.print(
            f"  [yellow]⚠ Index/chunk mismatch:[/yellow] "
            f"{n_vectors} vectors vs {n_chunks} chunks — rebuilding."
        )
        return None
 
    logfire.info(
        "Index loaded from cache",
        vectors=n_vectors,
        chunks=n_chunks,
        index_path=str(INDEX_FILE),
    )
    console.print(
        f"  [green]✓ Index loaded from cache[/green] "
        f"({n_vectors} vectors, {n_chunks} chunks)"
    )
    return index, chunks  
 
def _index_is_fresh(corpus_dir: Path) -> bool:
    """True if the saved corpus hash matches the current corpus files."""
    if not CORPUS_HASH_FILE.exists():
        logfire.warn("No corpus hash file found", expected_path=str(CORPUS_HASH_FILE))
        console.print(
            "  [yellow]⚠ No corpus fingerprint found[/yellow] "
            "— treating index as stale."
        )
        return False
    saved   = CORPUS_HASH_FILE.read_text().strip()
    current = _corpus_hash(corpus_dir)
    if saved != current:
        logfire.info(
            "Corpus has changed since last index build",
            saved_hash=saved[:12] + "...",
            current_hash=current[:12] + "...",
        )
        console.print(
            "  [yellow]⚠ Corpus fingerprint changed[/yellow] "
            "— index will be rebuilt."
        )
        return False
    return True
 
 
@logfire.instrument("build_faiss_index")
def build_faiss_index(
    docs:         list[Document],
    valid_chunks: list[ValidatedChunk],
    embeddings:   MistralAIEmbeddings,
    corpus_dir:   Path,
    batch_size:   int = 32,
    force_rebuild: bool = False,
) -> tuple[FAISS, list[ValidatedChunk]]:
    """
    Build (or reload) a FAISS index using true cosine similarity.
 
    Strategy
    --------
    1. If a valid cached index exists AND the corpus hasn't changed → reload.
    2. Otherwise: embed in batches, normalize, build IndexFlatIP, save.
 
    The index uses IndexFlatIP on L2-normalized vectors, which is
    mathematically equivalent to cosine similarity
    """
    import faiss as faiss_lib
    from langchain_community.docstore.in_memory import InMemoryDocstore

    # ── Embed in batches ─────────────────────────────────────────────────
    texts = [doc.page_content for doc in docs]
    all_embeddings: list[list[float]] = []
 
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch = texts[i:i + batch_size]
        with logfire.span("embed_batch", start=i, size=len(batch)):
            vecs = embeddings.embed_documents(batch)
            all_embeddings.extend(vecs)
 
    matrix = np.array(all_embeddings, dtype="float32")
 
    # ── Validate count ───────────────────────────────────────────────────
    if matrix.shape[0] != len(docs):
        msg = f"Embedding count mismatch: expected {len(docs)}, got {matrix.shape[0]}"
        logfire.error(msg)
        raise ValueError(msg)
 
    # ── Normalize → cosine via IndexFlatIP ───────────────────────────────
    faiss_lib.normalize_L2(matrix)
    dimension = matrix.shape[1]
    raw_index = faiss_lib.IndexFlatIP(dimension)
    raw_index.add(matrix)
 
    # ── Wrap into LangChain FAISS ─────────────────────────────────────────
    index_to_docstore_id = {i: str(i) for i in range(len(docs))}
    docstore = InMemoryDocstore({str(i): docs[i] for i in range(len(docs))})
    lc_index = FAISS(
        embedding_function   = embeddings,
        index                = raw_index,
        docstore             = docstore,
        index_to_docstore_id = index_to_docstore_id,
    )
 
    logfire.info(
        "FAISS index built",
        vectors   = raw_index.ntotal,
        dimension = dimension,
        metric    = "cosine (IndexFlatIP + L2 norm)",
    )
 
    # ── Persist for future runs ───────────────────────────────────────────
    _save_index(lc_index, valid_chunks, corpus_dir)
    return lc_index, valid_chunks

def run_indexing(
        input_directory: str,
        force_rebuild: bool = False
    )-> tuple[FAISS, list[ValidatedChunk]]:

    t0 = time.perf_counter()
    console.rule("[bold]NBA Reddit RAG — Indexing Pipeline[/bold]")
    # ── Validate config ──────────────────────────────────────────────────
    with logfire.span("validate_config"):
        config = PipelineConfig()
        console.print(f"[cyan]Config:[/cyan] {config.model_dump()}")
    # ── embedding model ──────────────────────────────────────────────────
    with logfire.span("init_generator_models"):
        mistral_embeddings = MistralAIEmbeddings(
            model   = config.embed_model,
            api_key = MISTRAL_API_KEY,
        )
        console.print(
            f"[cyan]Generator:[/cyan] {config.mistral_model} "
            f"| embeddings: {config.embed_model}"
        )   
    # ── Cache hit ────────────────────────────────────────────────────────
    if not force_rebuild and _index_is_fresh(CORPUS_DIR):
        index, chunks = _load_index(mistral_embeddings)
        if index is not None and chunks is not None:
            #logfire.info("Corpus not changed, index loaded from disk ",
            #        vectors=index.index.ntotal,
            #        chunks=len(chunks))
            return index, chunks
    
    logfire.info("Building index from scratch")
    console.print("  [yellow]Building index from scratch …[/yellow]")
    # ── Load corpus ───────────────────────────────────────────────────────
    console.print("\n[bold]Step 1/3[/bold] Loading corpus …")
    from utils.load_data import load_corpus
    raw_docs = load_corpus(CORPUS_DIR)
 
    # ── Chunk & validate ──────────────────────────────────────────────────
    console.print("[bold]Step 2/3[/bold] Chunking & validating …")
    valid_chunks, lc_docs = chunk_and_validate(raw_docs, config)
    console.print(f"  → {len(valid_chunks)} valid chunks")
 
    # ── Build FAISS index (or reload from cache) ──────────────────────────
    # Built using Mistral embeddings — must match the embeddings used at
    # query time in CosineRetriever (same model family, same vector space).
    console.print("[bold]Step 3/3[/bold] Building FAISS index …")
    index, valid_chunks = build_faiss_index(
        docs          = lc_docs,
        valid_chunks  = valid_chunks,
        embeddings    = mistral_embeddings,
        corpus_dir    = CORPUS_DIR,
        batch_size    = config.batch_size,
        force_rebuild = force_rebuild,
    )
    elapsed = time.perf_counter() - t0
    logfire.info("Pipeline complete", elapsed_s=round(elapsed, 2))
    console.print(f"\n[bold green]Done in {elapsed:.1f}s[/bold green]")
    return index, valid_chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indexation for NBA Reddit RAG application ")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=INPUT_DIR,
        help=f"Directory containing the source files (default: {INPUT_DIR})"
    )
    parser.add_argument(
        "--force_rebuild",
        type=bool,
        default=False,
        help="Forcing rebuild Faiss index even if no input file has changed  (default: False)"
    )
    args = parser.parse_args()

    run_indexing(input_directory=args.input_dir, force_rebuild=args.force_rebuild)