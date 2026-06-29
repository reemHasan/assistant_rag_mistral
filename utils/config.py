# utils/config.py
import os
from dotenv import load_dotenv
from pathlib import Path
# Charger les variables d'environnement du fichier .env
load_dotenv()

# --- Clé API ---
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LOGFIRE_TOKEN = os.getenv("LOGFIRE_TOKEN")
if not MISTRAL_API_KEY:
    print("⚠️ Attention: La clé API Mistral (MISTRAL_API_KEY) n'est pas définie dans le fichier .env")
    # Vous pouvez choisir de lever une exception ici ou de continuer avec des fonctionnalités limitées
    # raise ValueError("Clé API Mistral manquante. Veuillez la définir dans le fichier .env")
elif not GEMINI_API_KEY:
        print("⚠️ Attention: La clé API Gemini (GEMINI_API_KEY) n'est pas définie dans le fichier .env")
elif not LOGFIRE_TOKEN:
        print("⚠️ Attention: Le token de Logfire (LOGFIRE_TOKEN) n'est pas définie dans le fichier .env")

# --- Modèles Mistral ---
EMBEDDING_MODEL = "mistral-embed-2312"
MODEL_NAME = "mistral-small-2506" # Ou un autre modèle comme mistral-large-latest
SQL_MODEL = "mistral-large-latest"
EVALUATION_MODEL_MISTRAL = "mistral-large-latest"

# --- Configuration de l'Indexation ---
# INPUT_DATA_URL = os.getenv("INPUT_DATA_URL") # Décommentez si vous utilisez une URL
BASE_DIR = Path(__file__).parent.parent
INPUT_DIR = BASE_DIR/"inputs/inputs_rag_tool"                # Dossier pour les données sources après extraction
VECTOR_DB_DIR = BASE_DIR/"vector_db"         # Dossier pour stocker l'index Faiss et les chunks
#FAISS_INDEX_FILE = os.path.join(VECTOR_DB_DIR, "faiss_index.idx")
#DOCUMENT_CHUNKS_FILE = os.path.join(VECTOR_DB_DIR, "document_chunks.pkl")
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".csv", ".xlsx", ".xls", ".docx"}
CHUNK_SIZE = 1500                   # Taille des chunks en *caractères* (vise ~512 tokens)
CHUNK_OVERLAP = 150                 # Chevauchement en *caractères*
EMBEDDING_BATCH_SIZE = 32           # Taille des lots pour l'API d'embedding

# --- Configuration de la Recherche ---
SEARCH_K = 5                        # Nombre de documents à récupérer par défaut

# --- Gemini Evaluation model & embedding -------
EVALUATION_MODEL_NAME = "gemini-3.1-flash-lite"
EVALUATION_EMBEDDING  = "gemini-embedding-2-preview"

# --- Configuration de la Base de Données ---
DATABASE_DIR = BASE_DIR/"sqlite_db"
DATABASE_FILE = os.path.join(DATABASE_DIR, "nba.db")
DATABASE_URL = f"sqlite:///{DATABASE_FILE}" # URL pour SQLAlchemy

# --- Configuration de l'Application ---
APP_TITLE = "NBA Analyst AI"
NAME = "NBA" # Nom à personnaliser dans l'interface