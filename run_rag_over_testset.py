# ───────────────────────────────────────────────────────────────────────────
#  RAG retrieval + generation
# ───────────────────────────────────────────────────────────────────────────
import json
import time
import logging
from pathlib import Path
import argparse
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
#import sys
#ROOT_DIR = Path(__file__).resolve().parent.parent
#sys.path.append(str(ROOT_DIR))
from utils.config import (MISTRAL_API_KEY, MODEL_NAME, SEARCH_K)
from utils.vector_store import VectorStoreManager


# --- Configuration du Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

# --- Configuration de l'API Mistral ---
api_key = MISTRAL_API_KEY
model = MODEL_NAME

if not api_key:
    logging.error("Erreur : Clé API Mistral non trouvée (MISTRAL_API_KEY). Veuillez la définir dans le fichier .env.")

try:
    client = MistralClient(api_key=api_key)
    logging.info("Client Mistral initialisé.")
except Exception as e:
    logging.exception(f"Erreur lors de l'initialisation du client Mistral : {e}")
    
def get_vector_store_manager()-> VectorStoreManager|None:
    logging.info("Tentative de chargement du VectorStoreManager...")
    try:
        manager = VectorStoreManager()
        # Vérifie si l'index a bien été chargé par le constructeur
        if manager.index is None or not manager.document_chunks:
            logging.error("Index Faiss ou chunks non trouvés/chargés par VectorStoreManager.")
            return None # Retourne None si échec
        logging.info(f"VectorStoreManager chargé avec succès ({manager.index.ntotal} vecteurs).")
        return manager
    except FileNotFoundError:
         logging.error("FileNotFoundError lors de l'init de VectorStoreManager.")
         return None
    except Exception as e:
        logging.exception(f"Erreur inattendue lors du chargement du VectorStoreManager: {e}")
        return None

def generate_reponse(prompt_messages: list[str]) -> str:
    """
    Envoie le prompt (qui inclut maintenant le contexte) à l'API Mistral.
    """
    if not prompt_messages:
         logging.warning("Tentative de génération de réponse avec un prompt vide.")
         return "Je ne peux pas traiter une demande vide."
    try:
        logging.info(f"Appel à l'API Mistral modèle '{model}' avec {len(prompt_messages)} message(s).")
        # Log le contenu du prompt (peut être long) - commenter si trop verbeux
        # logging.debug(f"Prompt envoyé à l'API: {prompt_messages}")

        response = client.chat(
            model=model,
            messages=prompt_messages,
            temperature=0.1, # Température basse pour des réponses factuelles basées sur le contexte
            # top_p=0.9,
        )
        if response.choices and len(response.choices) > 0:
            logging.info("Réponse reçue de l'API Mistral.")
            return response.choices[0].message.content
        else:
            logging.warning("L'API n'a pas retourné de choix valide.")
            return "Désolé, je n'ai pas pu générer de réponse valide pour le moment."
    except Exception as e:
        logging.exception(f"Erreur {e} API Mistral pendant client.chat")
        return "Je suis désolé, une erreur technique m'empêche de répondre. Veuillez réessayer plus tard."

def build_rag_answer(
    question:  str,
    index:     VectorStoreManager,
    top_k:     int,
) -> tuple[str, list[str]]:
    """Retrieve top-k chunks, call Mistral, return (answer, contexts)."""

    # 2. Vérifier si le Vector Store est disponible
    if index is None:
        logging.error("VectorStoreManager non disponible pour la recherche.")
        # On arrête ici car on ne peut pas faire de RAG
        return ("Index not found","Can not generate response")
     # 3. Rechercher le contexte dans le Vector Store
    try:
        logging.info(f"Recherche de contexte pour la question: '{question}' avec k={SEARCH_K}")
        retrieved = index.search(question, k=top_k)
        logging.info(f"{len(retrieved)} chunks trouvés dans le Vector Store.")
    except Exception as e:
        logging.exception(f"Erreur {e} pendant vector_store_manager.search pour la query: {question}")
    
    if not retrieved:
        context_block = "Aucune information pertinente trouvée dans la base de connaissances pour cette question."
        logging.warning(f"Aucun contexte trouvé pour la query: {question}")
    else:
        contexts  = [doc['text'] for doc in retrieved]
        context_block = "\n\n---\n\n".join(contexts)
    # --- Prompt Système pour RAG ---
    # Adaptez ce prompt selon vos besoins
    prompt = ("Tu es 'NBA Analyst AI', un assistant expert sur la ligue de basketball NBA. "
    "Ta mission est de répondre aux questions des fans en animant le débat."
    f"Context:\n{context_block}\n\n"
    f"QUESTION DU FAN: {question}\n\n"
    "RÉPONSE DE L'ANALYSTE NBA:"
    )
    message_mistral_api = [
        ChatMessage(role="user", content=prompt)
    ]
    answer = generate_reponse(message_mistral_api)
    return answer.strip(), contexts


def run_rag_over_testset(
    dataset_file: str= "eval_dataset.json",
    dataset_output_file: str ="eval_dataset_with_answers.json"
    ):
    """Fill answer + contexts for every TestRow."""
    # 1. Get the saved index 
    index = get_vector_store_manager()
    
    # 2. Load dataset
    with open(dataset_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # 3. Loop over questions
    for sample in dataset:
        question = sample["question"]
        # Call the RAG prototype
        try:
            answer, contexts = build_rag_answer(question, index, top_k=SEARCH_K)
            # Insert results back
            sample["answer"] = answer
            sample["contexts"] = contexts
            logging.info("New eval raw has been treated")
            time.sleep(30)
        except Exception as exc:
                logging.error(f"RAG step failed for question: {question} for error: {exc}")
# Save updated dataset
    with open(dataset_output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)



if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent
    parser = argparse.ArgumentParser(description="Create reponse&context for testset using rag prototype")
    parser.add_argument(
        "--testset_name",
        type=str,
        default="eval_dataset.json",
        help="The name of testset to be used in evaluation of rag assistant"
    )
    parser.add_argument(
        "--testset_filled",
        type=str,
        default="eval_dataset_with_answers.json",
        help="The name of testset after filled with reponse&context from rag assistant"
    )
    args = parser.parse_args()
    input_eval_dataset = BASE_DIR/f"evaluation/eval_artifacts/{args.testset_name}"
    output_eval_dataset = BASE_DIR/f"evaluation/eval_artifacts/{args.testset_filled}"
    run_rag_over_testset(input_eval_dataset, output_eval_dataset)