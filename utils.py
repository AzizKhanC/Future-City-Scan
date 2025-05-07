import os
import geopandas as gpd
import numpy as np
from rasterstats import zonal_stats
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from os.path import exists
from osgeo import gdal, osr
import numpy as np
from pathlib import Path


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def set_paths(base_dir):
    data = os.path.join(base_dir, 'data') 
    # output = os.path.join(base_dir, 'output') 
    output = Path(base_dir) / "02-process-output"
    # shapefiles = os.path.join(output, 'shapefiles') 
    shapefiles = os.path.join(f"{base_dir}/01-inputs", 'shapefiles') 
    maps = os.path.join(output, 'maps') 
    rasters = os.path.join(output, 'rasters') 
    tables = os.path.join(output, 'tables') 
    
    dirs_list= [data, output, base_dir, shapefiles , maps , rasters , tables]
    for dir in dirs_list:
        if not os.path.exists(dir):
            os.mkdir(dir)
    return data, shapefiles , maps , rasters , output ,tables


# For re-projecting input vector layer to raster projection 
def reproject_gpdf(input_gpdf, raster):
    proj = raster.crs.to_proj4()
    reproj = input_gpdf.to_crs(proj)
    return reproj

# For selecting which raster statistics to calculate
def list_statistics(stat):
    out_stats = stat
    return out_stats


# For calculating zonal statistics
def calculate_zonal_stats(vector, raster, stats):
    # Run zonal statistics, store result in geopandas dataframe
    result = zonal_stats(vector, raster, stats=stats, geojson_out=True)
    geostats = gpd.GeoDataFrame.from_features(result)
    for c in stats:
        geostats[c] = np.round(geostats[c] , decimals= 2)
    return geostats



# clipping the raster to the buffered AOI and exporting it. 
def clip_and_export_rescale_raster(input_gpdf, raster , out_raster, rescaling_factor):
    Vector=input_gpdf
    with rasterio.open(raster) as src:
        Vector=Vector.to_crs(src.crs)
        # print(Vector.crs)
        out_image, out_transform=mask(src,Vector.geometry,crop=True)
        no_data_value=0
        out_image = np.where(out_image<0,no_data_value,out_image) #replaced negative values with 
        out_image = out_image*rescaling_factor
        out_meta=src.meta.copy() # copy the metadata of the source DEM
        
    out_meta.update({
        "driver":"Gtiff",
        "height":out_image.shape[1], # height starts with shape[1]
        "width":out_image.shape[2], # width starts with shape[2]
        "transform":out_transform
    })
                
    with rasterio.open(out_raster,'w',**out_meta) as dst:
        dst.write(out_image)

# clipping the raster to the buffered AOI and exporting it. 
def clip_and_export_raster(input_gpdf, raster , out_raster):
    Vector=input_gpdf
    with rasterio.open(raster) as src:
        Vector=Vector.to_crs(src.crs)
        print(Vector.crs, src.crs)
        out_image, out_transform=mask(src,Vector.geometry,crop=True)
        no_data_value=0
        out_image = np.where(out_image<0,no_data_value,out_image) #replaced negative values with 
        out_meta=src.meta.copy() # copy the metadata of the source DEM
        print(1)
    out_meta.update({
        "driver":"Gtiff",
        "height":out_image.shape[1], # height starts with shape[1]
        "width":out_image.shape[2], # width starts with shape[2]
        "transform":out_transform
    })
    print(2)     
    with rasterio.open(out_raster,'w',**out_meta) as dst:
        dst.write(out_image)


def reproject_rasters(crs , unprojected_raster, projected_raster):
    if not exists(projected_raster):
        with rasterio.open(unprojected_raster) as src:
            dst_crs = 'EPSG:' + str(crs)
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds)
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': dst_crs,
                'transform': transform,
                'width': width,
                'height': height
            })

            with rasterio.open(projected_raster, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest)

# Updated



def reproject_and_clip_raster(input_raster, shapefile, output_raster, target_epsg, multiply_factor=1):
    """Reproject, clip, and apply a multiplication factor to each pixel in a raster, setting NoData and negative values to 0."""
    
    # Open the input raster
    src = gdal.Open(input_raster)
    
    # Get the source CRS and define the target CRS
    src_proj = osr.SpatialReference(wkt=src.GetProjection())
    target_proj = osr.SpatialReference()
    target_proj.ImportFromEPSG(target_epsg)
    
    # Step 1: Reproject and clip the raster with gdal.Warp
    temp_raster = '/vsimem/temp_reprojected.tif'  # In-memory temporary file
    # gdal.Warp(
    #     temp_raster,
    #     src,
    #     format="GTiff",
    #     cutlineDSName=shapefile,         # Shapefile for clipping boundary
    #     cropToCutline=True,              # Crop to the shapefile's bounding box
    #     dstSRS=target_proj.ExportToWkt() # Set target projection without alpha band
    # )
    
    gdal.Warp( 
    temp_raster,
    src,
    format="GTiff",
    cutlineDSName=shapefile,        # Shapefile for clipping boundary
    cropToCutline=True,             # Crop to the shapefile's bounding box
    # cutlineBlend=0.0,               # Prevents feathering along the edges
    dstSRS=target_proj.ExportToWkt(), # Set target projection without alpha band
    warpOptions=["CUTLINE_ALL_TOUCHED=TRUE"] # Ensures that features touching the cutline are included
    )
    
    # Step 2: Multiply each pixel by the given factor, setting NoData and negative values to 0
    temp_ds = gdal.Open(temp_raster)
    
    # Create the output raster file with the same dimensions and geotransform as temp_ds
    driver = gdal.GetDriverByName("GTiff")
    output_ds = driver.Create(
        output_raster,
        temp_ds.RasterXSize,
        temp_ds.RasterYSize,
        1,  # Ensure a single band
        temp_ds.GetRasterBand(1).DataType
    )
    output_ds.SetGeoTransform(temp_ds.GetGeoTransform())
    output_ds.SetProjection(temp_ds.GetProjection())
    
    # Process the first band
    band = temp_ds.GetRasterBand(1)
    data = band.ReadAsArray()  # Read data as an array
    
    # Set NoData values and negative values to 0
    nodata_value = band.GetNoDataValue()
    if nodata_value is not None:
        data[data == nodata_value] = 0
    data[data < 0] = 0
    
    # Apply the multiplication factor
    data = data * multiply_factor
    
    # Write the modified data to the output raster
    output_band = output_ds.GetRasterBand(1)
    output_band.WriteArray(data)
    
    # Set the NoData value to 0 in the output raster
    output_band.SetNoDataValue(0)
    
    # Clean up
    src = None
    temp_ds = None
    output_ds = None
    gdal.Unlink(temp_raster)  # Remove in-memory temporary file




def raster_sum_mean(raster_path):
    # Open the raster file
    print(f"Opening: {raster_path}")
    dataset = gdal.Open(raster_path)
    if dataset is None:
        raise FileNotFoundError(f"Cannot open {raster_path}")
    
    # Read the first raster band as array
    band = dataset.GetRasterBand(1)
    array = band.ReadAsArray()
    
    # Calculate the sum and mean, ignoring no-data values
    nodata_value = band.GetNoDataValue()
    if nodata_value is not None:
        array = np.where(array == nodata_value, np.nan, array)
    
    # Compute sum and mean
    total_sum = np.nansum(array)
    mean_value = np.nanmean(array)
    
    return total_sum, mean_value

