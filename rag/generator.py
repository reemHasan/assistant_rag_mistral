import logging
from pathlib import Path
import pprint
import logfire
from rich.console import Console
from mistralai import Mistral as MistralClient
from .retriever import CosineRetriever
from langchain_mistralai import MistralAIEmbeddings

# ───────────────────────────────────────────────────────────────────────────
#  Bootstrap
# ───────────────────────────────────────────────────────────────────────────
console = Console()
logging.basicConfig(level=logging.WARNING)
#logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))
import sys
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
from utils.config import  MISTRAL_API_KEY
from utils.models import PipelineConfig

# ───────────────────────────────────────────────────────────────────────────
# RAG retrieval + generation
# ───────────────────────────────────────────────────────────────────────────
@logfire.instrument("build_rag_answer")
def build_rag_answer(
    question:    str,
    retriever:   CosineRetriever,
    mistral_llm: MistralClient,
    model_name:  str,
    top_k:       int,
) -> tuple[str, list[str]]:
    """
    Retrieve top-k chunks via normalized cosine search, call Mistral
    using the raw SDK v1.x (mistralai==1.9.10).

    v1.x API changes vs v0.x:
      - MistralClient  → Mistral  (imported as MistralClient alias above)
      - client.chat()  → client.chat.complete()
      - ChatMessage()  → plain dict {"role": ..., "content": ...}
    """
    try:
        retrieved = retriever.retrieve(query=question, k=top_k)
        contexts  = [doc.page_content for doc in retrieved]

        context_block = "\n\n---\n\n".join(contexts)
        prompt = (
            "You are an NBA analytics assistant for coaches. "
            "Answer the question strictly using the context below. "
            "If the answer is not in the context, say 'Information not available in the source.'\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {question}\n\nAnswer:"
        )
        # v1.x: chat.complete() with plain dict messages (no ChatMessage class)
        response = mistral_llm.chat.complete(
            model    = model_name,
            messages = [{"role": "user", "content": prompt}],
        )
        answer = response.choices[0].message.content
        return answer.strip(), contexts
    except  Exception as e:
            logfire.error(f"Error while searching for answer: {e}")
            return []

# ───────────────────────────────────────────────────────────────────────────
#  Main orchestration
# ───────────────────────────────────────────────────────────────────────────
 
def main() -> None:
    """RAG-powered conversational NBA agent with Mistral small"""
    # ── Validate config ──────────────────────────────────────────────────
    with logfire.span("validate_config"):
        config = PipelineConfig()
        console.print(f"[cyan]Config:[/cyan] {config.model_dump()}")
 
    # ── Generator models ──────────────────────────────────────────────────
    # mistral_llm       : raw Mistral SDK v1.x — used in build_rag_answer
    # mistral_embeddings: LangChain wrapper — used in build_faiss_index
    #                     and CosineRetriever (embed_documents / embed_query)
    with logfire.span("init_generator_models"):
        mistral_llm = MistralClient(api_key=MISTRAL_API_KEY)
        mistral_embeddings = MistralAIEmbeddings(
            model   = config.embed_model,
            api_key = MISTRAL_API_KEY,
        )
        console.print(
            f"[cyan]Generator:[/cyan] {config.mistral_model} "
            f"| embeddings: {config.embed_model}"
        )

    # ── RAG inference (Mistral generator answers using retrieved context) ──
    retriever = CosineRetriever(embeddings=mistral_embeddings)
    print("\n" + "="*70)
    print("🤖 NBA Reddit RAG AGENT")
    print("="*70)
    print("💬 Ask me about NBA teams or matches using natural language!")
    print("\nExample queries:")
    print("  • 'Which player has the highest 3-point percentage?'")
    print("  • 'Which team has the most rebounds?'")
    print("\nCommands:")
    print("  • 'quit' - Exit the chatbot")
    print("-" * 70)
        
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                print("🤖 Bot: Please ask me any question about NBA !!")
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n🤖 Bot: Thank you for using the RAG Reddit RAG AGENT!")
                print("      Hope you found some interesting information! 👋")
                break
            else:
                # Process the user query with RAG Agent
                response = build_rag_answer(user_input, retriever, mistral_llm, config.mistral_model, config.top_k)
                pprint.pprint(response[0])

        except KeyboardInterrupt:
            print("\n\n🤖 Bot: Goodbye! Hope you find what you looking for! 👋")
            break
        except Exception as e:
            print(f"❌ Bot: Sorry, I encountered an error: {e}")
    

if __name__ == "__main__":
    main()


