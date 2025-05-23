import panel as pn
from models.anndata_model import AnnDataModel
from models.chat_assistant import ChatAssistant
from views.main_view import MainView

def main():
    # Set Panel theme
    pn.extension(sizing_mode="stretch_width")
    
    # Initialize models
    anndata_model = AnnDataModel()

    # Initialize chat assistant with data_controller's query_cells method
    chat_assistant = ChatAssistant("", anndata_model)
    
    # Create main view
    main_view = MainView(anndata_model, chat_assistant)
    
    # Load a predetermined dataset
    data_path = "/home/wojciech/private/parda_v2/tests/data/small.h5ad"
    try:
        anndata_model.load_data(data_path)
        anndata_model.calculate_scvi_embeddings()
        anndata_model.calculate_clip_embeddings()
        anndata_model.prepare_additional_columns()
        main_view.update_plot()
    except Exception as e:
        print(f"Error loading data: {str(e)}")
    
    # Serve the application
    return main_view.panel().servable()

if __name__ == "__main__":
    main().show()