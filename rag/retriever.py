# ───────────────────────────────────────────────────────────────────────────
#  Cosine-correct retriever
# ───────────────────────────────────────────────────────────────────────────
import logfire
import numpy as np
from rich.console import Console
from langchain_mistralai import  MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
#import sys
#ROOT_DIR = Path(__file__).resolve().parent.parent
#sys.path.append(str(ROOT_DIR))
from rag.indexer import _load_index

console = Console()

class CosineRetriever:
    """
    Wraps a LangChain FAISS index to guarantee true cosine similarity
    by normalizing the query vector before every search.
 
    Also handles index loading from disk in __init__ — since the retriever
    is always needed for inference, it is the natural place to own the
    "load_index" logic.
 
    Loading behaviour
    ─────────────────
    If index=None is passed and a saved index exists on disk, __init__
    loads it automatically via _load_index().  If no saved index exists
    AND no index object was provided, a RuntimeError is raised immediately
 
    Why query normalization matters
    ─────────────────────────────────
    The FAISS index was built with:
        faiss.normalize_L2(matrix)   # stored vectors are unit-length
        IndexFlatIP.add(matrix)      # inner product on unit vectors = cosine
 
    LangChain's similarity_search() passes the raw embed_query() vector
    to FAISS without normalizing it — inner product between a unit stored
    vector and a non-unit query vector is NOT cosine similarity.
    This class intercepts the search, normalizes first, then calls
    FAISS's lower-level index.search() directly.
    """
 
    def __init__(
        self,
        embeddings: MistralAIEmbeddings,
        index:      FAISS | None = None,
    ) -> None:
        self.embeddings = embeddings
 
        if index is not None:
            # Caller already has a built/loaded index — use it directly
            self.index = index
            logfire.info(
                "CosineRetriever initialised with provided index",
                vectors=index.index.ntotal,
            )
        else:
            # No index provided — try loading from disk
            console.print(
                "  [cyan]CosineRetriever:[/cyan] no index provided, "
                "attempting to load from disk …"
            )
            loaded = _load_index(embeddings)
            if loaded is None:
                # _load_index already printed which file was missing/corrupt
                raise RuntimeError(
                    "CosineRetriever: could not load a FAISS index from disk "
                    "Run build_faiss_index() first, or pass a pre-built index."
                )
            self.index, self.chunks = loaded
            logfire.info(
                "CosineRetriever: index loaded from disk in __init__",
                vectors=self.index.index.ntotal,
                chunks=len(self.chunks),
            )
            #console.print( f"CosineRetriever: index loaded from disk in __init__ with {len(self.chunks)} " )

    def retrieve(
        self,
        query: str,
        k:     int = 4,
    ) -> list[Document]:
        """
        Embed → normalize → search.
        Returns the top-k Documents with cosine_score in metadata.
        """
        import faiss as faiss_lib
        try:
            # 1. Embed the query
            raw_vec = self.embeddings.embed_query(query)
    
            # 2. Normalize to unit length — matches index-time normalization
            vec = np.array([raw_vec], dtype="float32")   # shape (1, dim)
            faiss_lib.normalize_L2(vec)          
    
            # 3. Search directly on the raw FAISS index (bypasses LangChain path)
            scores, indices = self.index.index.search(vec, k)
    
            # 4. Resolve integer indices → Documents via LangChain docstore
            if indices.size > 0: # verify if there are results
                docs: list[Document] = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx == -1:
                        continue
                    doc_id = self.index.index_to_docstore_id.get(idx)
                    if doc_id is None:
                        continue
                    doc = self.index.docstore.search(doc_id)
                    if doc is None:
                        continue
                    doc.metadata["cosine_score"] = round(float(score), 4)
                    doc.metadata["similarity_score"] = score * 100
                    docs.append(doc)
                # Sort by score (highest similarity first)
                docs.sort(key=lambda x: x.metadata["cosine_score"], reverse=True)
                logfire.info(
                    "Cosine retrieval",
                    query_snippet = query[:60],
                    k             = k,
                    returned      = len(docs),
                    top_score     = round(float(scores[0][0]), 4) if len(scores[0]) else None,
                )
                console.print( f"CosineRetriever: found {len(docs)} interesting documents for the current question " )
            else:
                logfire.warning(f"No results found for this question {query}")
                console.print(f"  [yellow]⚠ No results found for this question {query}[/yellow] ")
            return docs
        
        except Exception as e:
            logfire.error(f"Error while searching documents for the question: {e}")
            return []
