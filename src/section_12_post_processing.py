import os
import glob
from osgeo import gdal, ogr

def polygonize_rasters(input_dir, output_dir, start_keyword, end_keyword):
    """
    Function to loop through rasters, polygonize them, and export as GeoJSON with polygon IDs, pixel values, and filename fields.

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
        
        # Split the filename to extract the last two components
        name_parts = base_name.split("_")
        field1 = name_parts[-2] if len(name_parts) > 1 else ""
        field2 = name_parts[-1] if len(name_parts) > 0 else ""

        # Clean up field names by removing invalid characters
        invalid_chars = ".''"
        field1 = ''.join(c for c in field1 if c not in invalid_chars)
        field2 = ''.join(c for c in field2 if c not in invalid_chars)

        # Open the raster dataset
        raster_ds = gdal.Open(raster_file)
        if raster_ds is None:
            print(f"Error opening raster file: {raster_file}")
            continue

        # Create output GeoJSON file path
        output_geojson = os.path.join(output_dir, f"{base_name}.geojson")

        # Create a memory layer for the vectorized data
        driver = ogr.GetDriverByName("GeoJSON")
        vector_ds = driver.CreateDataSource(output_geojson)
        layer = vector_ds.CreateLayer(base_name, srs=None, geom_type=ogr.wkbPolygon)

        # Add fields for polygon IDs, pixel values, and filename components
        field_defn_id = ogr.FieldDefn("PolygonID", ogr.OFTInteger)
        layer.CreateField(field_defn_id)
        field_defn_value = ogr.FieldDefn("PixelValue", ogr.OFTReal)
        layer.CreateField(field_defn_value)
        field_defn_field1 = ogr.FieldDefn("Field1", ogr.OFTString)
        layer.CreateField(field_defn_field1)
        field_defn_field2 = ogr.FieldDefn("Field2", ogr.OFTString)
        layer.CreateField(field_defn_field2)

        # Get the band (assuming single band raster)
        band = raster_ds.GetRasterBand(1)
        if band is None:
            print(f"Error: No band found in raster file: {raster_file}")
            continue

        # Polygonize raster to vector
        gdal.Polygonize(
            band,  # Input raster band
            None,  # Mask band (use None for no mask)
            layer,  # Output layer
            -1,  # Field index (not used directly here)
            [],  # Options (none in this case)
            callback=None  # Progress callback
        )

        # Assign IDs, pixel values, and filename components to polygons
        for i, feature in enumerate(layer):
            feature.SetField("PolygonID", i + 1)
            # Get the pixel value from the raster for the polygon geometry
            geometry = feature.GetGeometryRef()
            if geometry is not None:
                # Compute the centroid of the polygon to sample the raster value
                centroid = geometry.Centroid()
                px = int(centroid.GetX())
                py = int(centroid.GetY())
                pixel_value = band.ReadAsArray(px, py, 1, 1)
                if pixel_value is not None:
                    feature.SetField("PixelValue", float(pixel_value[0, 0]))
            # Set the filename components as fields
            feature.SetField("Field1", field1)
            feature.SetField("Field2", field2)
            layer.SetFeature(feature)

        # Close datasets
        vector_ds = None
        raster_ds = None

        print(f"Polygonized {raster_file} to {output_geojson}")

