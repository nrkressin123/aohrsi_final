import rasterio
from rasterio.mask import mask
import geopandas as gpd

# Load data
raster_path = "data/basel_img1.tif"
vector_path = "data/predicted_blds_regularized.geojson"
output_path = "data/basel_img1_masked_raster.tif"

with rasterio.open(raster_path) as src:
    gdf = gpd.read_file(vector_path)
    geoms = [geom.__geo_interface__ for geom in gdf.geometry]

    # Mask the raster using the polygons
    out_image, out_transform = mask(src, geoms, crop=False, filled=True, nodata=0)

    # Update metadata for the output file
    out_meta = src.meta.copy()
    out_meta.update({
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform,
        "nodata": 0
    })

# Save masked raster
with rasterio.open(output_path, "w", **out_meta) as dest:
    dest.write(out_image)

print(f"Masked raster saved to {output_path}")