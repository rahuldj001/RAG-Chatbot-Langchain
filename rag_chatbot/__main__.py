"""Main executable CLI script for running the Local RAG Chatbot.

Initializes the environment, parses CLI flags, starts the background services
(if needed), instantiates the RAG pipeline, and launches the Gradio Web UI.
"""

import argparse
import os
import shutil
import huggingface_hub

# Compatibility patch: gradio 4.x expects `HfFolder` on `huggingface_hub`.
# Prevents import errors with recent versions of huggingface_hub.
if not hasattr(huggingface_hub, "HfFolder"):
    class _HfFolder:
        @classmethod
        def get_token(cls):
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

from dotenv import load_dotenv
from .ui import LocalChatbotUI
from .pipeline import LocalRAGPipeline
from .logger import Logger
from .ollama import run_ollama_server, is_port_open

# Load API keys and environment variables from .env
load_dotenv()

# CONSTANTS: Log paths, uploaded document storage, and avatar icons for chat
LOG_FILE = "logging.log"
DATA_DIR = "data/data"
AVATAR_IMAGES = ["./assets/user.png", "./assets/bot.png"]

# CLI ARGUMENT PARSER: Configures runtime options for network host, sharing, and model
parser = argparse.ArgumentParser(description="Run the RAG Chatbot Web UI")
parser.add_argument(
    "--host",
    type=str,
    default="localhost",
    help="Set host to 'localhost' or 'host.docker.internal' if running in Docker",
)
parser.add_argument(
    "--share",
    action="store_true",
    help="Create a public Gradio share link (e.g. via Gradio tunneling)",
)
parser.add_argument(
    "--model",
    type=str,
    default="openai/gpt-oss-20b",
    help="Default LLM model name (Groq API, OpenAI API, or local Ollama model)",
)
args = parser.parse_args()

# LOCAL OLLAMA SERVER MANAGEMENT:
# If running locally without an external cloud API key (Groq/OpenAI),
# checks whether local Ollama CLI is installed and automatically boots it on port 11434.
if args.host != "host.docker.internal":
    port_number = 11434
    if not os.environ.get("GROQ_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        if shutil.which("ollama") and not is_port_open(port_number):
            run_ollama_server()

# LOGGING SETUP:
# Resets the local log file
logger = Logger(LOG_FILE)
logger.reset_logs()

# RAG PIPELINE INITIALIZATION:
# Sets up embedding models, vector stores, chat engines, and LLM instances
pipeline = LocalRAGPipeline(host=args.host, default_model=args.model)

# UI CONSTRUCTION:
# Builds the Gradio Blocks interface with sidebar settings, chat area, and PDF file processing
ui = LocalChatbotUI(
    pipeline=pipeline,
    logger=logger,
    host=args.host,
    data_dir=DATA_DIR,
    avatar_images=AVATAR_IMAGES,
)

# LAUNCH THE WEB APPLICATION:
# Binds to local port 7860 and serves the web frontend
ui.build().launch(
    share=args.share,
    server_name="127.0.0.1",
    server_port=7860,
    debug=False,
    show_api=False,
)
