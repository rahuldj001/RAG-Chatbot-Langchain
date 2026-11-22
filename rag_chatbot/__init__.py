"""Package entry point for rag_chatbot.

Exposes primary high-level interfaces: LocalRAGPipeline and run_ollama_server.
Also applies compatibility patches for third-party libraries across Python versions.
"""

import huggingface_hub

# Compatibility patch: gradio 4.x expects `HfFolder` on `huggingface_hub`,
# but it was moved/deprecated in huggingface_hub >= 0.25.
# Providing a lightweight fallback prevents import crashes when starting the UI.
if not hasattr(huggingface_hub, "HfFolder"):
    class _HfFolder:
        @classmethod
        def get_token(cls):
            # Retrieve the Hugging Face authentication token if available
            return huggingface_hub.get_token()

        @classmethod
        def save_token(cls, token):
            pass

    huggingface_hub.HfFolder = _HfFolder

# Compatibility patch: gradio 4.x passes (name, context_dict) to Starlette's TemplateResponse,
# but Starlette >= 0.38 expects (request, name, context_dict).
try:
    import starlette.templating
    _orig_template_response = starlette.templating.Jinja2Templates.TemplateResponse

    def _patched_template_response(self, *args, **kwargs):
        if len(args) >= 2 and isinstance(args[0], str) and isinstance(args[1], dict):
            name = args[0]
            context = args[1]
            req = kwargs.pop("request", context.get("request"))
            return _orig_template_response(self, request=req, name=name, context=context, **kwargs)
        return _orig_template_response(self, *args, **kwargs)

    starlette.templating.Jinja2Templates.TemplateResponse = _patched_template_response
except ImportError:
    pass

# Export core pipeline and server runner for modular access
from .pipeline import LocalRAGPipeline
from .ollama import run_ollama_server

__all__ = [
    "LocalRAGPipeline",
    "run_ollama_server",
]
