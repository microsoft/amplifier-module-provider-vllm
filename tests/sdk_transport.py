"""Use the installed SDK's transport, even when both HTTP packages coexist."""
import importlib

import openai

httpx = importlib.import_module(
    "httpx2" if int(openai.__version__.split(".")[0]) >= 3 else "httpx"
)
