

## File details
### `data_module.py`
> This module serves as the central hub for all data preparation and management tasks within this project. Its primary function is to bridge the gap between raw geospatial files and the format required for PyTorch model training. \
The separation of data logic into `data_module.py` ensures that the main training script remains clean, modular, and focused solely on model definition and training loops.
#### Core responsibilities
|Component|	Function|	Key Technology|
|-|-|-|
|Raster Data Processing| Reads, converts, and preprocesses raw GeoTIFF (raster) files. This includes handling multiple spectral bands and managing NoData values.|	Rasterio and NumPy
|Dataset Creation| Defines the GeoDataset class, which handles indexing, loading, and returning individual data samples (image-label pairs).|PyTorch Dataset|
|Data Management|Defines the GeoDataModule class, which manages the complete lifecycle of the data, including splitting the files into Training, Validation, and Test sets.|PyTorch DataLoader|
|Batching|Provides DataLoader instances to efficiently load data in optimized batches to feed the GPU during the training process.|PyTorch DataLoader|