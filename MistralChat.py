# MistralChat.py (version Agent RAG & SQL)
import streamlit as st
import logging
from langchain_core.messages import HumanMessage, AIMessage
# --- Importations depuis vos modules ---
try:
    from utils.config import (
        MISTRAL_API_KEY, MODEL_NAME, DATABASE_URL,
        APP_TITLE, NAME
    )
    from utils.models import PipelineConfig
    from agent_react import build_agent
except ImportError as e:
    st.error(f"Erreur d'importation: {e}. Vérifiez la structure de vos dossiers .")
    st.stop()


# --- Configuration du Logging ---
# Note: Streamlit peut avoir sa propre gestion de logs. Configurer ici est une bonne pratique.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

# --- Configuration de l'API Mistral ---
api_key = MISTRAL_API_KEY
model = MODEL_NAME

if not api_key:
    st.error("Erreur : Clé API Mistral non trouvée (MISTRAL_API_KEY). Veuillez la définir dans le fichier .env.")
    st.stop()

#------- functions -----------------------
def history_to_text(history):
    lines = []

    for msg in history:
        if isinstance(msg, HumanMessage):
            lines.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            lines.append(f"Assistant: {msg.content}")

    return "\n".join(lines)

# --- Chargement du Vector Store (mis en cache) ---
@st.cache_resource # Garde l'agent chargé en mémoire pour la session
def get_agent():
    logging.info("Tentative de chargement du agent...")
    try:
        config = PipelineConfig() 
        agent = build_agent(db_uri=DATABASE_URL, mistral_api_key=MISTRAL_API_KEY, agent_model=config.mistral_model, 
                         sql_model=config.sql_generator_model, embed_model=config.embed_model, top_k_rag=config.top_k, verbose=True, chat=True)
        # Vérifie si l'agent a bien été chargé par le constructeur
        if agent is None:
            st.error("L'AgentExecuter n'est pas pu être chargés.")
            st.warning("Assurez-vous d'avoir exécuté 'python indexer.py' et 'load_excel_to_db.py' après avoir placé vos fichiers dans le dossier 'inputs'.")
            logging.error("AgentExecuter non chargé par build_agent")
            return None # Retourne None si échec
        logging.info("AgentExecuter chargé avec succès")
        return agent
    except Exception as e:
        st.error(f"Erreur inattendue lors du chargement du AgentExecuter: {e}")
        logging.exception("Erreur chargement AgentExecuter")
        return None

agent = get_agent()

# --- Initialisation de l'historique de conversation ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content=f"Bonjour ! Je suis votre analyste IA pour la {NAME}. Posez-moi vos questions sur les équipes, les joueurs ou les statistiques, et je vous répondrai en me basant sur les données les plus récentes.")
    ]

# --- Interface Utilisateur Streamlit ---
st.title(APP_TITLE)
st.caption(f"Assistant virtuel pour {NAME} | Modèle: {model}")

# Affichage des messages de l'historique (pour l'UI)
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)

# Zone de saisie utilisateur
if user_prompt := st.chat_input(f"Posez votre question sur la {NAME}..."):
    # 1. Ajouter et afficher le message de l'utilisateur
    #st.session_state.messages.append({"role": "user", "content": user_prompt})
    st.session_state.chat_history.append(HumanMessage(content=user_prompt))
    with st.chat_message("user"):
        st.write(user_prompt)

    # 2. Générer la réponse de l'assistant via LLM
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.text("...") # Indicateur simple
        try:
            result = agent.invoke(
                {"input": user_prompt,
                "chat_history": history_to_text(st.session_state.chat_history),}
                )
            response_content = result["output"]
            # Affichage de la réponse complète
            message_placeholder.write(response_content)
        except Exception as e:
            st.error(f"Erreur lors de l'appel à l'API Mistral: {e}")
            logging.exception("Erreur API Mistral pendant AgentExecuter")

    # 3. Ajouter la réponse de l'assistant à l'historique (pour affichage UI)
    #st.session_state.messages.append({"role": "assistant", "content": response_content})
    st.session_state.chat_history.append(AIMessage(content=response_content))

# Petit pied de page optionnel
st.markdown("---")
st.caption("Powered by LangChain & Mistral AI & Faiss | Data-driven NBA Insights")