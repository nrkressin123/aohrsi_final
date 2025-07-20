import geopandas as gpd
import pandas as pd

# Load your layers
truth = gpd.read_file('arcgis_data/test_data.shp')
pred = gpd.read_file('arcgis_data/predicted_data.shp')

# Ensure same CRS
pred = pred.to_crs(truth.crs)

# Spatial Intersection
inter = gpd.overlay(truth, pred, how='intersection')

# Examine the columns to find the correct field names
print(inter.columns)

## rename
inter = inter.rename(columns={
    'classvalue': 'Truth',
    'gridcode': 'Predicted'
})

# Calculate the geometry area (units depend on CRS, usually square meters if using UTM)
inter['area'] = inter.geometry.area

# Aggregate area for each (truth, predicted) combination
area_matrix = inter.pivot_table(
    index='Truth',
    columns='Predicted',
    values='area',
    aggfunc='sum',
    fill_value=0   # if there are missing combinations, set them to 0 area
)

print("Area (units of CRS) confusion matrix:")
print(area_matrix)