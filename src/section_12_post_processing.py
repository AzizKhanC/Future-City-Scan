import os
import glob
from osgeo import gdal, ogr
from pathlib import Path

import os
import glob
from osgeo import ogr

from shapely.geometry import shape
import geopandas as gpd
import os

import os
import glob
from osgeo import gdal, ogr



def polygonize_rasters(input_dir, output_dir, start_keyword, end_keyword):
    """
    Function to loop through rasters, polygonize them, and export as GeoJSON with accurate pixel values.

    Parameters:
        input_dir (str): Directory containing raster files.
        output_dir (str): Directory to save the output GeoJSON files.
        start_keyword (str): Keyword the filenames should start with.
        end_keyword (str): Keyword the filenames should end with.
    """
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Find all rasters matching the criteria
    pattern = os.path.join(input_dir, f"{start_keyword}*{end_keyword}")
    raster_files = glob.glob(pattern)

    for raster_file in raster_files:
        # Extract filename without extension
        base_name = os.path.splitext(os.path.basename(raster_file))[0]

        # Open the raster dataset
        raster_ds = gdal.Open(raster_file)
        if raster_ds is None:
            print(f"Error opening raster file: {raster_file}")
            continue

        # Get raster geotransform and band
        geotransform = raster_ds.GetGeoTransform()
        band = raster_ds.GetRasterBand(1)
        if band is None:
            print(f"Error: No band found in raster file: {raster_file}")
            raster_ds = None
            continue

        # Get NoData value
        nodata_value = band.GetNoDataValue()

        # Create output GeoJSON file path
        output_geojson = os.path.join(output_dir, f"{base_name}.geojson")

        # Create a memory layer for the vectorized data
        driver = ogr.GetDriverByName("GeoJSON")
        vector_ds = driver.CreateDataSource(output_geojson)
        layer = vector_ds.CreateLayer(base_name, srs=None, geom_type=ogr.wkbPolygon)

        # Add a field for pixel values
        field_defn_value = ogr.FieldDefn("PixelValue", ogr.OFTReal)
        layer.CreateField(field_defn_value)

        # Polygonize raster to vector
        temp_mem_vector = "/vsimem/temp_vector"
        mem_driver = ogr.GetDriverByName("Memory")
        mem_vector_ds = mem_driver.CreateDataSource(temp_mem_vector)
        mem_layer = mem_vector_ds.CreateLayer("temp_layer", geom_type=ogr.wkbPolygon)
        mem_layer.CreateField(field_defn_value)

        gdal.Polygonize(
            band,  # Input raster band
            None,  # Mask band (use None for no mask)
            mem_layer,  # Output memory layer
            0,  # Field index
            [],  # Options (none in this case)
            callback=None  # Progress callback
        )

        # Transfer features from memory layer to GeoJSON layer
        for feature in mem_layer:
            pixel_value = feature.GetField("PixelValue")
            if pixel_value is not None and pixel_value != nodata_value:
                new_feature = ogr.Feature(layer.GetLayerDefn())
                new_feature.SetGeometry(feature.GetGeometryRef().Clone())
                new_feature.SetField("PixelValue", pixel_value)
                layer.CreateFeature(new_feature)
                new_feature = None  # Prevent memory leaks

        # Close datasets
        mem_vector_ds = None
        vector_ds = None
        raster_ds = None

        print(f"Polygonized {raster_file} to {output_geojson}")


# def polygonize_rasters(input_dir, output_dir, start_keyword, end_keyword):
#     """
#     Function to loop through rasters, polygonize them, and export as GeoJSON with pixel values.

#     Parameters:
#         input_dir (str): Directory containing raster files.
#         output_dir (str): Directory to save the output GeoJSON files.
#         start_keyword (str): Keyword the filenames should start with.
#         end_keyword (str): Keyword the filenames should end with.
#     """
    
#     # Ensure the output directory exists
#     os.makedirs(output_dir, exist_ok=True)

#     # Find all rasters matching the criteria
#     pattern = os.path.join(input_dir, f"{start_keyword}*{end_keyword}")
#     raster_files = glob.glob(pattern)

#     for raster_file in raster_files:
#         # Extract filename without extension
#         base_name = os.path.splitext(os.path.basename(raster_file))[0]

#         # Open the raster dataset
#         raster_ds = gdal.Open(raster_file)
#         if raster_ds is None:
#             print(f"Error opening raster file: {raster_file}")
#             continue

#         # Get raster geotransform and band
#         geotransform = raster_ds.GetGeoTransform()
#         band = raster_ds.GetRasterBand(1)
#         if band is None:
#             print(f"Error: No band found in raster file: {raster_file}")
#             continue

#         # Create output GeoJSON file path
#         output_geojson = os.path.join(output_dir, f"{base_name}.geojson")

#         # Create a memory layer for the vectorized data
#         driver = ogr.GetDriverByName("GeoJSON")
#         vector_ds = driver.CreateDataSource(output_geojson)
#         layer = vector_ds.CreateLayer(base_name, srs=None, geom_type=ogr.wkbPolygon)

#         # Add a field for pixel values
#         field_defn_value = ogr.FieldDefn("PixelValue", ogr.OFTReal)
#         layer.CreateField(field_defn_value)

#         # Polygonize raster to vector
#         gdal.Polygonize(
#             band,  # Input raster band
#             None,  # Mask band (use None for no mask)
#             layer,  # Output layer
#             0,  # Field index
#             [],  # Options (none in this case)
#             callback=None  # Progress callback
#         )

#         # Assign pixel values to polygons
#         layer.ResetReading()  # Reset reading to start accessing features
#         feature = layer.GetNextFeature()
#         while feature:
#             geometry = feature.GetGeometryRef()
#             if geometry is not None:
#                 # Get centroid coordinates in raster space
#                 centroid = geometry.Centroid()
#                 x, y = centroid.GetX(), centroid.GetY()
#                 px = int((x - geotransform[0]) / geotransform[1])
#                 py = int((y - geotransform[3]) / geotransform[5])

#                 # Check if pixel coordinates are within raster bounds
#                 if 0 <= px < raster_ds.RasterXSize and 0 <= py < raster_ds.RasterYSize:
#                     pixel_value = band.ReadAsArray(px, py, 1, 1)
#                     if pixel_value is not None:
#                         value = pixel_value[0, 0]
#                         feature.SetField("PixelValue", float(value))
#                         layer.SetFeature(feature)
#             feature = layer.GetNextFeature()  # Move to the next feature

#         # Close datasets
#         vector_ds = None
#         raster_ds = None

#         print(f"Polygonized {raster_file} to {output_geojson}")




def add_fields_to_geojsons(geojson_dir, start_keyword, end_keyword): 
    """
    Function to add additional fields to GeoJSON files.

    Parameters:
        geojson_dir (str): Directory containing GeoJSON files.
        start_keyword (str): Keyword the filenames should start with.
        end_keyword (str): Keyword the filenames should end with.
    """

    # Find all GeoJSON files matching the criteria
    pattern = os.path.join(geojson_dir, f"{start_keyword}*{end_keyword}")
    geojson_files = glob.glob(pattern)
    print(f"Found {len(geojson_files)} files matching the pattern: {pattern}")

    for geojson_file in geojson_files:
        print(f"Processing file: {geojson_file}")
        
        # Extract filename without extension
        base_name = os.path.splitext(os.path.basename(geojson_file))[0]
        print(f"Base name extracted: {base_name}")

        # Extract year and scenario from filename
        name_parts = base_name.split("_")
        year = name_parts[-1] if len(name_parts) > 1 else "Unknown"
        scenario = name_parts[-2] if len(name_parts) > 2 else "Unknown"
        print(f"Year extracted: {year}, Scenario extracted: {scenario}")

        # Open the GeoJSON file
        driver = ogr.GetDriverByName("GeoJSON")
        vector_ds = driver.Open(geojson_file, 1)  # Open for writing
        if vector_ds is None:
            print(f"Error opening GeoJSON file: {geojson_file}")
            continue
        else:
            print(f"Successfully opened {geojson_file}")

        layer = vector_ds.GetLayer()

        # Add new fields: FileName, Year, Scenario
        if layer.FindFieldIndex("FileName", 0) == -1:
            print(f"Adding 'FileName' field to {geojson_file}")
            layer.CreateField(ogr.FieldDefn("FileName", ogr.OFTString))
        else:
            print(f"'FileName' field already exists in {geojson_file}")

        if layer.FindFieldIndex("Year", 0) == -1:
            print(f"Adding 'Year' field to {geojson_file}")
            layer.CreateField(ogr.FieldDefn("Year", ogr.OFTString))
        else:
            print(f"'Year' field already exists in {geojson_file}")

        if layer.FindFieldIndex("Scenario", 0) == -1:
            print(f"Adding 'Scenario' field to {geojson_file}")
            layer.CreateField(ogr.FieldDefn("Scenario", ogr.OFTString))
        else:
            print(f"'Scenario' field already exists in {geojson_file}")

        # Update features with new field values
        for feature in layer:
            feature.SetField("FileName", base_name)
            feature.SetField("Year", year)
            feature.SetField("Scenario", scenario)
            layer.SetFeature(feature)
        
        print(f"Updated features in {geojson_file}")

        # Close the dataset
        vector_ds = None
        print(f"Closed the GeoJSON file: {geojson_file}")

        print(f"Updated {geojson_file} with additional fields")




def append_geojsons(geojson_dir, start_keyword):
    """
    Function to append all GeoJSON files that start with a specific keyword,
    add an ID column for polygons, and return the combined features as a dictionary for debugging.

    Parameters:
        geojson_dir (str): Directory containing GeoJSON files.
        start_keyword (str): Keyword that the filenames should start with.

    Returns:
        list: A list of dictionaries representing the combined GeoJSON features.
    """
    # Ensure the directory exists
    if not os.path.exists(geojson_dir):
        print(f"Directory does not exist: {geojson_dir}")
        return []

    # Find all GeoJSON files that start with the specified keyword
    pattern = os.path.join(geojson_dir, f"{start_keyword}*.geojson")
    geojson_files = glob.glob(pattern)
    print(f"Found {len(geojson_files)} files matching the pattern: {pattern}")

    if not geojson_files:
        print("No GeoJSON files found.")
        return []

    # Create a GeoJSON driver
    driver = ogr.GetDriverByName("GeoJSON")
    if not driver:
        print("GeoJSON driver not available.")
        return []

    # Initialize variables
    combined_features = []
    feature_id = 1  # Unique feature ID

    # Process each GeoJSON file
    for geojson_file in geojson_files:
        print(f"Processing file: {geojson_file}")

        # Open the current GeoJSON file
        vector_ds = driver.Open(geojson_file, 0)
        if vector_ds is None:
            print(f"Error opening GeoJSON file: {geojson_file}")
            continue

        layer = vector_ds.GetLayer()
        print(f"Layer: {layer.GetName()} - Feature Count: {layer.GetFeatureCount()}")

        # Append features with ID field to the combined list
        for feature in layer:
            feature_json = feature.ExportToJson(as_object=True)  # Convert feature to JSON
            feature_json["properties"]["ID"] = feature_id  # Add unique ID to properties
            combined_features.append(feature_json)
            feature_id += 1

        print(f"Appended features from {geojson_file}")

    print(f"Total features combined: {len(combined_features)}")

    # Return the combined panel for debugging
    return combined_features




def export_to_geojson(panel, output_path):
    """
    Convert a dictionary-based panel with GeoJSON-like geometries to a GeoDataFrame
    and export it as a GeoJSON file.

    Parameters:
        panel (list): List of features where each feature is a dictionary containing
                      "geometry" (GeoJSON-like) and "properties".
        output_path (str): Path to export the combined GeoJSON file.

    Returns:
        gpd.GeoDataFrame: The resulting GeoDataFrame.
    """
    # Convert GeoJSON-like geometries to Shapely objects
    for feature in panel:
        feature["geometry"] = shape(feature["geometry"])

    # Create GeoDataFrame
    combined_gdf = gpd.GeoDataFrame(
        [feature["properties"] for feature in panel],  # Extract properties
        geometry=[feature["geometry"] for feature in panel]  # Extract geometries
    )

    # Export GeoDataFrame to GeoJSON
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Exporting GeoDataFrame to: {output_path}")
    try:
        combined_gdf.to_file(output_path, driver="GeoJSON")
        print("Export successful!")
    except Exception as e:
        print(f"Error exporting GeoJSON: {e}")

    return combined_gdf



import os
import shutil
from fnmatch import fnmatch

def copy_rasters_with_keyword(source_dir, destination_dir, keyword, file_extension="tif"):
    """
    Copies raster files with a specific keyword in their names from source_dir 
    (including subdirectories) to destination_dir.

    :param source_dir: Path to the source directory.
    :param destination_dir: Path to the destination directory.
    :param keyword: Keyword to search for in filenames.
    :param file_extension: File extension of raster files (default is "tif").
    """
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    
    # Loop through all directories and files in the source directory
    for root, _, files in os.walk(source_dir):
        for file in files:
            if keyword in file and fnmatch(file, f"*.{file_extension}"):
                source_path = os.path.join(root, file)
                destination_path = os.path.join(destination_dir, file)
                print(f"source_path- {source_path} ----- destination_path {destination_path}")
                shutil.copy2(source_path, destination_path)  # Preserve metadata
                print(f"Copied: {source_path} -> {destination_path}")
                
                

def clip_geojsons(input_folder, output_folder, shapefile_path, skip_keyword):
    """
    Clips all GeoJSON files in a folder using a shapefile.
    
    Parameters:
        input_folder (str): Path to the folder containing GeoJSON files.
        output_folder (str): Path to the folder to save clipped GeoJSON files.
        shapefile_path (str): Path to the shapefile used for clipping.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Open the clipping shapefile
    driver = ogr.GetDriverByName("ESRI Shapefile")
    shapefile = driver.Open(shapefile_path, 0)  # 0 means read-only
    if shapefile is None:
        print("Failed to open shapefile:", shapefile_path)
        return
    
    clip_layer = shapefile.GetLayer()
    
    # Iterate through GeoJSON files in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".geojson"):
            if skip_keyword not in file_name:
                input_file_path = os.path.join(input_folder, file_name)
                output_file_path = os.path.join(output_folder, file_name)
                
                # Open the input GeoJSON
                input_driver = ogr.GetDriverByName("GeoJSON")
                input_data = input_driver.Open(input_file_path, 0)  # 0 means read-only
                
                if input_data is None:
                    print(f"Failed to open {file_name}")
                    continue
                
                input_layer = input_data.GetLayer()
                
                # Set up the output GeoJSON
                if os.path.exists(output_file_path):
                    os.remove(output_file_path)  # Remove if it exists
                
                output_data = input_driver.CreateDataSource(output_file_path)
                output_layer = output_data.CreateLayer(
                    input_layer.GetName(),
                    geom_type=input_layer.GetGeomType(),
                    srs=input_layer.GetSpatialRef()
                )
                
                # Copy fields from input to output
                input_layer_def = input_layer.GetLayerDefn()
                for i in range(input_layer_def.GetFieldCount()):
                    field_def = input_layer_def.GetFieldDefn(i)
                    output_layer.CreateField(field_def)
                
                # Clip features
                for feature in input_layer:
                    geom = feature.GetGeometryRef()
                    if geom is None:
                        continue
                    
                    # Check intersection with the clipping layer
                    geom_clone = geom.Clone()
                    for clip_feature in clip_layer:
                        clip_geom = clip_feature.GetGeometryRef()
                        if clip_geom is None:
                            continue
                        if geom_clone.Intersects(clip_geom):
                            clipped_geom = geom_clone.Intersection(clip_geom)
                            new_feature = ogr.Feature(output_layer.GetLayerDefn())
                            new_feature.SetGeometry(clipped_geom)
                            
                            # Copy field values
                            for i in range(input_layer_def.GetFieldCount()):
                                new_feature.SetField(
                                    input_layer_def.GetFieldDefn(i).GetNameRef(),
                                    feature.GetField(i)
                                )
                            
                            output_layer.CreateFeature(new_feature)
                            new_feature = None  # Release the feature
                    
                    geom_clone = None  # Release geometry
                    
                # Cleanup
                input_data = None
                output_data = None
                print(f"Clipped {file_name} to {output_file_path}")
        
    print("Clipping completed!")
