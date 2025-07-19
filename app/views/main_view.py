import panel as pn
import param
import numpy as np
import pandas as pd
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, BoxZoomTool, WheelZoomTool, ResetTool, PanTool, LassoSelectTool, ColorBar
from bokeh.transform import factor_cmap, linear_cmap
from bokeh.palettes import Turbo256, Category20
from pandas.api.types import is_numeric_dtype

class MainView(param.Parameterized):
    selected_cells = param.List(default=[], doc="List of selected cell IDs")
    color_by = param.String(default=None, doc="Metadata column to color by")
    
    def __init__(self, anndata_model, chat_assistant, **params):
        super().__init__(**params)
        self.anndata_model = anndata_model
        self.chat_assistant = chat_assistant
        self.color_selectors = {}
        self.init_ui()
        
    def init_ui(self):
        # Create layout
        self.plot_panel = self.create_plot_panel()
        self.chat_panel = self.create_chat_panel()
        self.left_panel = self.create_metadata_panel()
        
        # Main layout
        self.main_layout = pn.Column(
            pn.Row(
                pn.Column("scRNA-seq Explorer with AI Assistant", 
                          sizing_mode='stretch_width', 
                          height=50),
                sizing_mode='stretch_width'
            ),
            pn.Row(
                self.left_panel,
                self.plot_panel,
                self.chat_panel,
                sizing_mode='stretch_both'
            ),
            sizing_mode='stretch_both'
        )
    
    def create_metadata_panel(self):
        return pn.Column(
            pn.pane.Markdown("## Metadata Columns"),
            sizing_mode='stretch_width',
            width=250
        )
    
    def create_plot_panel(self):
        # Create empty plot initially
        p = figure(width=600, height=600, tools="",
                  title="UMAP Visualization")
        
        # Add required tools
        p.add_tools(LassoSelectTool())
        p.add_tools(WheelZoomTool())
        p.add_tools(BoxZoomTool())
        p.add_tools(PanTool())
        p.add_tools(ResetTool())
        
        # Add hover tool
        hover = HoverTool(tooltips=[("Cell ID", "@cell_id")])
        p.add_tools(hover)
        
        # Empty data source initially
        self.source = ColumnDataSource(data=dict(x=[], y=[], cell_id=[]))
        
        # Add scatter points
        self.scatter = p.scatter('x', 'y', source=self.source, size=3, alpha=0.6)
        
        self.plot = p
        self.color_bar = None
        self.legend_panel = None
        
        # Create panel
        return pn.Column(
            f"0 cells selected",
            p,
            sizing_mode='stretch_both'
        )
    
    def create_chat_panel(self):
        # Chat interface text area (read-only)
        self.chat_history = pn.widgets.TextAreaInput(
            value="", height=400, disabled=True, sizing_mode='stretch_both',
            placeholder="Chat history will appear here..."
        )
        
        # Chat input field
        self.chat_input = pn.widgets.TextInput(
            placeholder="Type your message here...", sizing_mode='stretch_width'
        )
        
        # Send button
        self.send_button = pn.widgets.Button(
            name='Send', button_type='primary', sizing_mode='fixed',
            width=100
        )
        self.send_button.on_click(self.send_message)
        
        # API Key input
        self.api_key_input = pn.widgets.PasswordInput(
            name='OpenAI API Key', placeholder='Enter your OpenAI API key',
            sizing_mode='stretch_width'
        )
        
        chat_panel = pn.Column(
            pn.Row("Chat Assistant", sizing_mode='stretch_width'),
            self.chat_history,
            pn.Row(
                self.chat_input,
                self.send_button,
                sizing_mode='stretch_width'
            ),
            self.api_key_input,
            sizing_mode='stretch_both',
            width=350
        )
        
        return chat_panel
    
    def send_message(self, event=None):
        if not self.chat_input.value:
            return
            
        if not self.api_key_input.value:
            self.update_chat_history("Please enter your OpenAI API key first.")
            return
            
        # Initialize chat assistant if not done yet
        if not self.chat_assistant.is_ready():
            self.chat_assistant.set_api_key(self.api_key_input.value)
        
        # Send message and get response
        user_message = self.chat_input.value
        self.chat_input.value = ""
        self.chat_assistant.send_message(
            user_message, 
            self.selected_cells if self.selected_cells else None,
            self.update_chat_history,
            self.highlight_cells,
            self.query_cells
        )
    
    def update_chat_history(self, text):
        self.chat_history.value = text
    
    def update_metadata_panel(self):
        if self.anndata_model.adata is None:
            return
        
        obs_df = self.anndata_model.adata.obs
        
        # Clear existing panels
        self.left_panel.clear()
        self.left_panel.append(pn.pane.Markdown("## Metadata Columns"))
        self.color_selectors = {}
        
        # For each metadata column
        for col in obs_df.columns:
            if len(obs_df[col].unique()) >= 50 and not is_numeric_dtype(obs_df[col]):
                continue
            col_panel = pn.Column(sizing_mode='stretch_width')
            # Create color selector
            color_selector = pn.widgets.Checkbox(name="", value=False)
            color_selector.param.watch(lambda event, col_name=col: self.set_color_by(col_name if event.new else None), 'value')
            self.color_selectors[col] = color_selector
            
            # Add header with toggle
            col_panel.append(pn.Row(pn.pane.Markdown(f"**{col}**"), color_selector))
            self.left_panel.append(col_panel)
    
    def set_color_by(self, col_name):
        # Reset all other color selectors
        for col, selector in self.color_selectors.items():
            if col != col_name:
                selector.value = False
        self.color_by = col_name
        self.update_colors()


    def highlight_cells(self, query):
        self.anndata_model.highlight_cells(query)
        self.set_color_by("marked")
        self.color_selectors["marked"].value = True

    def query_cells(self, query):
        result = self.anndata_model.query_cells(query)
        self.set_color_by("queried")
        self.color_selectors["queried"].value = True
        return result

    
    def create_category_legend(self, col_name, factors, palette):
        legend_items = []
        for i, factor in enumerate(factors):
            color = palette[i % len(palette)]
            legend_items.append(
                pn.Row(
                    pn.pane.HTML(f'<div style="width:20px; height:20px; background-color:{color}; border: 1px solid black;"></div>'), 
                    pn.pane.Markdown(f"{factor}"),
                    width=200
                )
            )
        
        # Create a grid layout with multiple columns if there are many items
        num_cols = max(1, min(3, len(factors) // 10 + 1))
        rows = []
        current_row = []
        
        for i, item in enumerate(legend_items):
            current_row.append(item)
            if (i + 1) % num_cols == 0 or i == len(legend_items) - 1:
                rows.append(pn.Row(*current_row))
                current_row = []
        
        return pn.Column(
            pn.pane.Markdown(f"## Legend: {col_name}"),
            *rows,
            margin=5,
            css_classes=['legend-panel']
        )
    
    def update_colors(self):
        # Clear existing legend panel
        if self.legend_panel is not None:
            if self.legend_panel in self.plot_panel:
                self.plot_panel.remove(self.legend_panel)
            self.legend_panel = None
        
        # Reset to default if no color selected
        if self.anndata_model.adata is None or self.color_by is None:
            # Reset to default color
            self.scatter.glyph.fill_color = "navy"
            self.scatter.glyph.line_color = "navy"
            
            # Remove color bar if exists
            if self.color_bar is not None:
                self.plot.right.remove(self.color_bar)
                self.color_bar = None
            
            return
        
        # Get the data for coloring
        col_data = self.anndata_model.adata.obs[self.color_by]
        
        # Update source data
        self.source.data[self.color_by] = col_data.astype(str).values
        
        # Remove old color bar if exists
        if self.color_bar is not None:
            self.plot.right.remove(self.color_bar)
            self.color_bar = None
        
        # Different coloring for categorical vs continuous data
        if not is_numeric_dtype(col_data):
            # Categorical coloring
            factors = list(col_data.unique())
            n_factors = len(factors)
            if n_factors <= 20:
                palette = Category20[20][:n_factors]
            else:
                palette = Turbo256[:n_factors]
            
            color_mapper = factor_cmap(self.color_by, palette=palette, factors=factors)
            self.scatter.glyph.fill_color = color_mapper
            self.scatter.glyph.line_color = color_mapper
            
            # Create and add legend panel
            self.legend_panel = self.create_category_legend(self.color_by, factors, palette)
            self.plot_panel.append(self.legend_panel)
        else:
            # Continuous coloring
            low = col_data.min()
            high = col_data.max()
            color_mapper = linear_cmap(self.color_by, Turbo256, low, high)
            self.scatter.glyph.fill_color = color_mapper
            self.scatter.glyph.line_color = color_mapper
            self.color_bar = ColorBar(color_mapper=color_mapper['transform'], width=8)
            self.plot.add_layout(self.color_bar, 'right')
    
    def update_plot(self):
        if self.anndata_model.adata is None:
            return
            
        # Extract UMAP coordinates
        if 'X_umap' not in self.anndata_model.adata.obsm:
            import scanpy as sc
            sc.pp.neighbors(self.anndata_model.adata, use_rep='X_scvi')
            sc.tl.umap(self.anndata_model.adata)
            
        umap_coords = self.anndata_model.adata.obsm['X_umap']
        
        # Update data source
        self.source.data = {
            'x': umap_coords[:, 0],
            'y': umap_coords[:, 1],
            'cell_id': self.anndata_model.adata.obs_names.tolist()
        }
        
        # Update metadata panel with columns from the loaded data
        self.update_metadata_panel()
        
        # Setup selection callback
        def selection_callback(attr, old, new):
            print(f"Selection callback called with new: {new}")
            adata = self.anndata_model.adata

            adata.obs['selected'] = False

            if new:
                self.selected_cells = [self.source.data['cell_id'][i] for i in new]
                # Mark selected
                adata.obs.loc[self.selected_cells, 'selected'] = True
                self.plot_panel[0] = f"{len(self.selected_cells)} cells selected"
            else:
                self.selected_cells = []
                self.plot_panel[0] = "0 cells selected"
        
        self.source.selected.on_change('indices', selection_callback)
    
    def panel(self):
        return self.main_layout 