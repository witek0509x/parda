import panel as pn
import os
from pathlib import Path

import dotenv

# Load environment variables from .env if present
dotenv.load_dotenv()

from models.anndata_model import AnnDataModel
from models.chat_assistant import ChatAssistant
from views.main_view import MainView

def main():
    # Set Panel theme
    pn.extension(sizing_mode="stretch_width")
    
    # Initialize data model
    anndata_model = AnnDataModel()

    # Load a predetermined dataset BEFORE creating the chat assistant so that
    # embeddings, clustering and metadata summaries are ready for the system
    # prompt.
    data_path = "/home/wojciech/private/parda_v2/tests/data/E2f7-knockout.h5ad"
    try:
        anndata_model.load_data(data_path)
        anndata_model.calculate_scvi_embeddings()
        anndata_model.calculate_clip_embeddings()
        anndata_model.prepare_additional_columns()
    except Exception as e:
        print(f"Error loading data: {str(e)}")

    # Determine OpenAI key from env
    openai_key = os.getenv("OPENAI_KEY", "")

    # Create chat assistant, pass key automatically if available
    chat_assistant = ChatAssistant(openai_key, anndata_model)

    # Create main view
    main_view = MainView(anndata_model, chat_assistant)

    # Autofill API key field in UI if available
    if openai_key:
        try:
            main_view.api_key_input.value = openai_key
        except Exception:
            pass

    # Update plot after view creation
    try:
        main_view.update_plot()
    except Exception as e:
        print(f"Error plotting data: {str(e)}")
    
    # Serve the application
    return main_view.panel().servable()

if __name__ == "__main__":
    main().show()