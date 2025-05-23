# UMAP Visualization Coloring Feature

## Overview

This document describes the coloring functionality implemented in the scRNA-seq Explorer application. The feature allows users to color the UMAP visualization based on any metadata column available in the AnnData object, enhancing the ability to identify patterns and relationships in the single-cell data.

## Implementation Details

### Metadata Column Selection

- A list of available metadata columns is displayed in a side panel
- Each column has a checkbox that toggles coloring by that attribute
- Columns with excessive unique values (≥50) that aren't numeric are excluded to prevent performance issues
- When a column is selected, all other selections are automatically deselected

### Coloring Mechanisms

The application handles two types of metadata variables differently:

#### Categorical Variables
- Used when a column has fewer than 50 unique values
- Each category is assigned a distinct color from a carefully selected palette:
  - For ≤20 categories: Uses the Category20 palette
  - For >20 categories: Uses the Turbo256 palette
- A legend panel is displayed below the plot showing each category and its corresponding color
- The legend is organized in a responsive grid layout

#### Continuous Variables
- Used for numeric columns and columns with ≥50 unique values
- Values are mapped to a color gradient using the Turbo256 palette
- A color bar is displayed on the right side of the plot showing the range

### User Interface Features

- Clear visual feedback for the current coloring selection
- Automated cleanup of previous legends/color bars when switching between variables
- Consistent styling of legend elements with clear color boxes
- Hover tooltips are updated to show the value of the currently selected coloring variable

### Technical Implementation

- Uses Bokeh's `factor_cmap` for categorical variables
- Uses Bokeh's `linear_cmap` for continuous variables
- Legend panel built with Panel components for maximum flexibility
- Cleanup logic ensures that only one legend or color bar is shown at a time

## Usage

1. Load a dataset into the application
2. The metadata columns will automatically populate in the left panel
3. Click the checkbox next to a column name to color the UMAP by that attribute
4. For categorical variables, refer to the legend below the plot
5. For continuous variables, use the color bar on the right side of the plot
6. Click the checkbox again to return to the default coloring 