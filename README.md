# Assistant RAG avec Mistral

Ce projet implémente un assistant virtuel basé sur le modèle Mistral, utilisant la technique de Retrieval-Augmented Generation (RAG) pour fournir des réponses précises et contextuelles à partir d'une base de connaissances personnalisée.

## Fonctionnalités

- 🔍 **Recherche sémantique** avec FAISS pour trouver les documents pertinents
- 🤖 **Génération de réponses** avec les modèles Mistral (Small ou Large)
- ⚙️ **Paramètres personnalisables** (modèle, nombre de documents, score minimum)

## Prérequis

- Python 3.12
- Clé API Mistral (obtenue sur [console.mistral.ai](https://console.mistral.ai/))

## Installation

1. **Cloner le dépôt**
```bash
git clone <url-du-repo>
cd <nom-du-repo>
```

2. **Créer un environnement virtuel avec uv**

```bash
# Python 3.12
python --version
# uv package manager
pip install uv
# Activation de l'environnement virtuel
# Sur Windows
venv\Scripts\activate
# Sur macOS/Linux
source venv/bin/activate
```
3. **Installer les dépendances**
```bash
uv sync
```
4. **Configurer la clé API**

Créez un fichier `.env` à la racine du projet avec le contenu suivant :

```
MISTRAL_API_KEY=votre_clé_api_mistral
```

## Structure du projet

```
.
├── MistralChat.py          # Application Streamlit principale
├── indexer.py              # Script pour indexer les documents
├── inputs/                 # Dossier pour les documents sources
├── vector_db/              # Dossier pour l'index FAISS et les chunks
├── database/               # Base de données SQLite pour les interactions
└── utils/                  # Modules utilitaires
    ├── config.py           # Configuration de l'application
    ├── database.py         # Gestion de la base de données
    └── vector_store.py     # Gestion de l'index vectoriel

```

## Utilisation

### 1. Ajouter des documents

Placez vos documents dans le dossier `inputs/`. Les formats supportés sont :
- PDF
- TXT
- DOCX
- CSV
- JSON

Vous pouvez organiser vos documents dans des sous-dossiers pour une meilleure organisation.

### 2. Indexer les documents

Exécutez le script d'indexation pour traiter les documents et créer l'index FAISS :

```bash
python indexer.py
```

Ce script va :
1. Charger les documents depuis le dossier `inputs/`
2. Découper les documents en chunks
3. Générer des embeddings avec Mistral
4. Créer un index FAISS pour la recherche sémantique
5. Sauvegarder l'index et les chunks dans le dossier `vector_db/`

### 3. Lancer l'application

```bash
streamlit run MistralChat.py
```

L'application sera accessible à l'adresse http://localhost:8501 dans votre navigateur.


## Modules principaux

### `utils/vector_store.py`

Gère l'index vectoriel FAISS et la recherche sémantique :
- Chargement et découpage des documents
- Génération des embeddings avec Mistral
- Création et interrogation de l'index FAISS

### `utils/query_classifier.py`

Détermine si une requête nécessite une recherche RAG :
- Analyse des mots-clés
- Classification avec le modèle Mistral
- Détection des questions spécifiques vs générales

### `utils/database.py`

Gère la base de données SQLite pour les interactions :
- Enregistrement des questions et réponses
- Stockage des feedbacks utilisateurs
- Récupération des statistiques

## Personnalisation

Vous pouvez personnaliser l'application en modifiant les paramètres dans `utils/config.py` :
- Modèles Mistral utilisés
- Taille des chunks et chevauchement
- Nombre de documents par défaut
- Nom de la commune ou organisation

