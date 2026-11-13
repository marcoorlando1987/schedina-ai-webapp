"""Entrypoint di compatibilità.

Le app già pubblicate su Streamlit Cloud con file principale `schedina_ai_app.py`
continuano a funzionare: questo file esegue semplicemente `app.py`.
"""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).with_name("app.py")), run_name="__main__")
