# helpers/llm_finder.py
from typing import Optional, List, TYPE_CHECKING
from langchain_core.language_models import BaseChatModel
from core.logger import logger

if TYPE_CHECKING:
    from helpers.config import Settings


class LLMFinder:
    """
    Explicit two-method LLM finder:
    1. connect_via_api()   - Uses a user-provided API key.
    2. connect_local()     - Discovers local Ollama models and asks the user to choose.
    """

    # ------------------------------------------------------------
    # Method 1: API-based connection (User provides key)
    # ------------------------------------------------------------
    @staticmethod
    def connect_via_api(
        api_key: str,
        model_name: str,
        temperature: float = 0.0,
        base_url: Optional[str] = None
    ) -> Optional[BaseChatModel]:
        """
        Connects to an API-based LLM (e.g., OpenAI) using the user's API key.

        Args:
            api_key: The user's API key (required).
            model_name: The model to use (default: "gpt-3.5-turbo").
            temperature: Sampling temperature (default: 3.0 per your spec).
            base_url: Optional custom base URL (for Azure or OpenAI-compatible proxies).

        Returns:
            A LangChain BaseChatModel, or None if connection fails.
        """
        if not api_key or not api_key.strip():
            logger.error("API key is required for connect_via_api().")
            return None

        # IMPORTANT: OpenAI's temperature range is 0.0 to 2.0.
        # Since you specified default 3.0, we clamp it to 2.0 and log a warning.
        if temperature > 2.0:
            logger.warning(
                f"Temperature {temperature} exceeds OpenAI's max (2.0). "
                "Clamping to 2.0 to avoid an API error."
            )
            temperature = 2.0

        try:
            from langchain_openai import ChatOpenAI

            logger.info(f"Connecting to API LLM: {model_name}")
            return ChatOpenAI(
                model=model_name,
                api_key=api_key.strip(),
                temperature=temperature,
                base_url=base_url,  # Optional: for Azure/OpenAI-compatible endpoints
            )
        except Exception as e:
            logger.error(f"Failed to connect via API: {e}")
            return None

    # ------------------------------------------------------------
    # Method 2: Local discovery + Interactive selection
    # ------------------------------------------------------------
    @staticmethod
    def discover_local_models() -> List[str]:
        """
        Scans the local machine for available Ollama models.

        Returns:
            A list of model names (e.g., ['llama3:latest', 'mistral:7b']).
            Returns an empty list if Ollama is not installed or not running.
        """
        try:
            import ollama
            response = ollama.list()
            models = response.get("models", [])
            model_names = [m.get("name") for m in models if m.get("name")]
            logger.info(f"Discovered {len(model_names)} local Ollama models.")
            return model_names
        except ImportError:
            logger.error("Ollama Python library not installed. Run: pip install ollama")
            return []
        except Exception as e:
            logger.error(f"Failed to discover local models (is Ollama running?): {e}")
            return []

    @staticmethod
    def connect_local(
        temperature: float = 0.0
    ) -> Optional[BaseChatModel]:
        """
        Discovers local models, lists them, and asks the user to choose one.

        Args:
            temperature: Sampling temperature (default: 3.0 per your spec).

        Returns:
            A LangChain ChatOllama instance, or None if no model is selected/discovered.
        """
        models = LLMFinder.discover_local_models()

        if not models:
            logger.error(
                "No local models found. Please install Ollama and pull a model "
                "(e.g., `ollama pull llama3`)."
            )
            return None

        # --- Step 2: List the models ---
        print("\n" + "=" * 50)
        print("Available Local LLMs (Ollama):")
        for idx, model_name in enumerate(models):
            print(f"  [{idx}] {model_name}")
        print("=" * 50)

        # --- Step 3: Ask the user to choose ---
        try:
            choice = input("Enter the number of the model you want to use: ").strip()
            if not choice.isdigit():
                raise ValueError("Invalid input. Please enter a number.")

            choice_idx = int(choice)
            if choice_idx < 0 or choice_idx >= len(models):
                raise IndexError(f"Selection out of range (0-{len(models)-1}).")

            selected_model = models[choice_idx]
            logger.info(f"User selected local model: {selected_model}")

            from langchain_ollama import ChatOllama

            # Ollama temperature typically works best between 0.0 and 1.0,
            # but we pass the user's value (default 3.0) as requested.
            return ChatOllama(
                model=selected_model,
                temperature=temperature,
            )

        except (ValueError, IndexError) as e:
            logger.error(f"Selection failed: {e}")
            return None
        except KeyboardInterrupt:
            logger.warning("User cancelled the selection.")
            return None

    @staticmethod
    def connect_from_settings(settings: "Settings") -> Optional[BaseChatModel]:
        """Create a non-interactive LLM client suitable for the web API."""
        provider = settings.AI_PROVIDER.strip().lower()

        if provider == "openai":
            if not settings.API_KEY or not settings.LLM_MODEL:
                logger.error("OpenAI requires API_KEY and LLM_MODEL in .env.")
                return None
            return LLMFinder.connect_via_api(
                api_key=settings.API_KEY,
                model_name=settings.LLM_MODEL,
                temperature=settings.LLM_TEMPERATURE,
                base_url=settings.LLM_BASE_URL,
            )

        if provider == "ollama":
            if not settings.LLM_MODEL:
                logger.error("Ollama requires LLM_MODEL in .env.")
                return None
            try:
                from langchain_ollama import ChatOllama
                return ChatOllama(
                    model=settings.LLM_MODEL,
                    temperature=settings.LLM_TEMPERATURE,
                    base_url=settings.LLM_BASE_URL,
                )
            except Exception as error:
                logger.error(f"Failed to connect to Ollama: {error}")
                return None

        logger.error("AI_PROVIDER must be either 'openai' or 'ollama'.")
        return None



if __name__ == "__main__":
    finder = LLMFinder()

    # Example usage: Connect via API
    api_key = input("Enter your OpenAI API key (or leave blank to skip): ").strip()
    if api_key:
        llm_api = finder.connect_via_api(api_key=api_key)
        if llm_api:
            print(f"Successfully connected to API LLM: {llm_api.model}")
        else:
            print("Failed to connect via API.")

    # Example usage: Connect to local Ollama model
    llm_local = finder.connect_local()
    if llm_local:
        print(f"Successfully connected to local LLM: {llm_local.model}")
