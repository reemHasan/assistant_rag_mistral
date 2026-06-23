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

        # 1. Embed the query
        raw_vec = self.embeddings.embed_query(query)
 
        # 2. Normalize to unit length — matches index-time normalization
        vec = np.array([raw_vec], dtype="float32")   # shape (1, dim)
        faiss_lib.normalize_L2(vec)          
 
        # 3. Search directly on the raw FAISS index (bypasses LangChain path)
        scores, indices = self.index.index.search(vec, k)
 
        # 4. Resolve integer indices → Documents via LangChain docstore
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
            docs.append(doc)
 
        logfire.info(
            "Cosine retrieval",
            query_snippet = query[:60],
            k             = k,
            returned      = len(docs),
            top_score     = round(float(scores[0][0]), 4) if len(scores[0]) else None,
        )
        return docs

"""    def search(self, query_text: str, k: int = 5, min_score: float = None) -> List[Dict[str, any]]:
        
        Recherche les k chunks les plus pertinents pour une requête.

        Args:
            query_text: Texte de la requête
            k: Nombre de résultats à retourner
            min_score: Score minimum (entre 0 et 1) pour inclure un résultat

        Returns:
            Liste des chunks pertinents avec leurs scores
        
        if self.index is None or not self.document_chunks:
            logging.warning("Recherche impossible: l'index Faiss n'est pas chargé ou est vide.")
            return []
        if not MISTRAL_API_KEY:
             logging.error("Recherche impossible: MISTRAL_API_KEY manquante pour générer l'embedding de la requête.")
             return []

        logging.info(f"Recherche des {k} chunks les plus pertinents pour: '{query_text}'")
        try:
            # 1. Générer l'embedding de la requête
            response = self.mistral_client.embeddings(
                model=EMBEDDING_MODEL,
                input=[query_text] # La requête doit être une liste
            )
            query_embedding = np.array([response.data[0].embedding]).astype('float32')

            # Normaliser l'embedding de la requête pour la similarité cosinus
            faiss.normalize_L2(query_embedding)

            # 2. Rechercher dans l'index Faiss
            # Pour IndexFlatIP: scores = produit scalaire (plus grand = meilleur)
            # indices: index des chunks correspondants dans self.document_chunks
            # Demander plus de résultats si un score minimum est spécifié
            search_k = k * 3 if min_score is not None else k
            scores, indices = self.index.search(query_embedding, search_k)

            # 3. Formater les résultats
            results = []
            if indices.size > 0: # Vérifier s'il y a des résultats
                for i, idx in enumerate(indices[0]):
                    if 0 <= idx < len(self.document_chunks): # Vérifier la validité de l'index
                        chunk = self.document_chunks[idx]
                        # Convertir le score en similarité (0-1)
                        # Pour IndexFlatIP avec vecteurs normalisés, le score est déjà entre -1 et 1
                        # On le convertit en pourcentage (0-100%)
                        raw_score = float(scores[0][i])
                        similarity = raw_score * 100

                        # Filtrer les résultats en fonction du score minimum
                        # Le min_score est entre 0 et 1, mais similarity est en pourcentage (0-100)
                        min_score_percent = min_score * 100 if min_score is not None else 0
                        if min_score is not None and similarity < min_score_percent:
                            logging.debug(f"Document filtré (score {similarity:.2f}% < minimum {min_score_percent:.2f}%)")
                            continue

                        results.append({
                            "score": similarity, # Score de similarité en pourcentage
                            "raw_score": raw_score, # Score brut pour débogage
                            "text": chunk["text"],
                            "metadata": chunk["metadata"] # Contient source, category, chunk_id_in_doc, start_index etc.
                        })
                    else:
                        logging.warning(f"Index Faiss {idx} hors limites (taille des chunks: {len(self.document_chunks)}).")

            # Trier par score (similarité la plus élevée en premier)
            results.sort(key=lambda x: x["score"], reverse=True)

            # Limiter au nombre demandé (k) si nécessaire
            if len(results) > k:
                results = results[:k]

            if min_score is not None:
                min_score_percent = min_score * 100
                logging.info(f"{len(results)} chunks pertinents trouvés (score minimum: {min_score_percent:.2f}%).")
            else:
                logging.info(f"{len(results)} chunks pertinents trouvés.")

            return results

        except MistralAPIException as e:
            logging.error(f"Erreur API Mistral lors de la génération de l'embedding de la requête: {e}")
            logging.error(f"  Détails: Status Code={e.status_code}, Message={e.message}")
            return []
        except Exception as e:
            logging.error(f"Erreur inattendue lors de la recherche: {e}")
            return []"""