# aohrsi_final
# Green Roof Detection in Basel, Switzerland Using Deep Learning and GIS
Project Overview

This project aims to identify **green roofs** in the city of **Basel, Switzerland**, using a combination of **semantic segmentation (DeepLabV3+), to identify building footprints, raster processing**, and **classification in ArcGIS Pro**. We applied deep learning techniques to isolate roof structures from aerial imagery and then classified those roof areas as either vegetated (green roofs) or non-vegetated. The accuracy of the classification was assessed using a spatial confusion matrix.

Workflow Summary
1. Input Data:
   - A high-resolution aerial image of Basel.
   - Vector dataset of building footprints (for training).
   - Ground-truth green roof classification data (for validation).
2. Semantic Segmentation (DeepLabV3+):
   - Trained a DeepLabV3+ model on Basel imagery to segment building roofs.
   - Created a binary mask identifying rooftop areas.
   - Exported results as polygons (GeoJSON).
3. Raster Masking:
   - Applied the predicted building polygons to the input image.
   - Generated a masked raster showing only roof areas.
   - This raster was used in ArcGIS Pro for green roof classification.
4. ArcGIS Pro Classification:
   - Used spectral and spatial properties to classify green roofs.
   - Output: vector shapefile containing green vs. non-green roof labels.
5. Accuracy Assessment:
   - Compared predicted classification against ground truth.
   - Computed a spatial confusion matrix based on area.
   - Measured performance with accuracy, precision, and IoU metrics.
<img width="1470" height="836" alt="diagram-export-22-07-2025-14_31_17" src="https://github.com/user-attachments/assets/5f323711-6039-43aa-83d1-1eeaf3a17ccc" />

## Detailed Explanation of Scripts


###  ‘deeplab.py’ — Semantic Segmentation
- Trains a DeepLabV3+ model (ResNet101 backbone) on aerial image patches.
- Generates a binary roof mask, evaluates using accuracy, IoU, Dice, and Boundary IoU*.
- Produces polygon features representing predicted roof footprints.
- Saves regularized output polygons and an evaluation report.

Within the green roof identification workflow, this script should be ran first. To run this script, first create a 'data' folder wherever your data directory is. Then, ensure that you a satellite image file titled 'basel_img1.tif' within your data directory. Further, for the model to read your training data, you also require a file titled 'building_footprints_img1.json' within the data directory. The script will predict building footprints based off of the satellite image and training data. The data files we used are available in [Sciebo](https://uni-muenster.sciebo.de/s/JY9GKtNxNJ9s5Cw). The script will output 2 files: a predicted building footprints file titled 'predicted_blds.geojson' and a regularized version titled 'predicted_blds_regularized.geojson', both into the 'data' folder. 
  
*To check how well the model performed, we use several metrics: accuracy tells us how many pixels were correctly labeled; IoU checks how much the predicted rooftops overlap with the real ones;
Dice is similar but a bit more forgiving; and Boundary IoU focuses on how well the edges of the roofs were captured. These help evaluate the precision of the segmentation.


###  ‘mask_raster_with_buildings.py’ — Building-Only Raster Creation
- Uses predicted building polygons to mask the input image.
- Outputs a raster containing **only rooftop pixels**, setting other areas to 0.
- Used as input in **ArcGIS Pro** for the green roof classification task.

This script is ran second in the workflow. It requires the 'predicted_blds_regularized.geojson' file produced in the first script, and the satellite image file we also used in the first script. It simply extracts the raster values that fall within the predicted building polygons. It outputs a file titled 'basel_img1_masked_raster.tif' into the data directory. 

###  ‘accuracy_assessment.py’ — Classification Evaluation
- Loads `truth` and `predicted` shapefiles from ArcGIS classification.
- Intersects them and computes an **area-based confusion matrix**.
- Results show how much area was correctly/incorrectly classified.

This script is ran last in the workflow, after green roof identification has been performed in ArcGIS Pro. It requires two files in the 'arcgis_data' directory: a file titled 'test_data.shp', and a file titled 'predicted_data.shp'. It will then evaluate the accuracy of the predicted data versus the test data using an area intersection. The script will output a confusion matrix into the Python console. 

## Detailed Explanation of ArcgisPro Model

Through the Image classification wizard tool in arcgis pro which runs a supervised object based random trees classifier. The following images show the parameters used.
We then classified which rooftops were green using ArcGIS Pro’s Classification Wizard. However, one major limitation is that ArcGIS doesn’t let you save the trained classification model — meaning it can’t be reused on other images. Because of this, we couldn’t apply our model to a separate test image. Instead, we performed a train/test split within the same image, dividing the labeled data to simulate a proper accuracy assessment. This allowed us to evaluate the classifier’s performance even though the model couldn’t be saved or deployed.

<img width="240" height="450" alt="1" src="https://github.com/user-attachments/assets/ca77207e-3b84-4d00-b69d-642b94c2481c" />
<img width="240" height="1000" alt="2" src="https://github.com/user-attachments/assets/763c4250-e91e-4cb5-8e20-296c7a26b6c8" />
<img width="240" height="1000" alt="3" src="https://github.com/user-attachments/assets/412e7c0a-f588-42ea-b956-317c0ecf5255" />
<img width="240" height="1000" alt="4" src="https://github.com/user-attachments/assets/dc12abe4-6da1-4eb6-b5c9-9e77b9c5be2a" />

##  Key Performance Metrics

| Metric               | Value (Example) |
|----------------------|------------------|
| Training Accuracy     | ~94%             |
| Validation Accuracy   | ~91%             |
| Mean IoU              | ~0.85            |
| Dice Coefficient      | ~0.89            |
| Boundary IoU          | ~0.83            |
> *(Actual valuesreported in `segmentation_report.txt`)*

##  Tools and Libraries
- **Deep Learning**: PyTorch, torchvision, DeepLabV3+
- **Geospatial Processing**: GeoPandas, Rasterio, TorchGeo
- **Classification**: ArcGIS Pro
- **Evaluation**: scikit-learn, NumPy, Matplotlib

##  Notes for Reproducibility
- Use a GPU-enabled environment (e.g. Colab, local CUDA GPU).
- Ensure all spatial data uses the **same CRS** for overlays and masking.
- Regularize polygons to improve classification and visualization.
- ArcGIS classification should be supervised with visual inspection for best results.
##  Potential Improvements
- Fine-tune DeepLabV3+ with more training samples.
- Incorporate NDVI or NIR bands to better detect vegetated surfaces.
- Apply transfer learning to scale the model to other cities.
- Use a model that allows trained data to be saved.

