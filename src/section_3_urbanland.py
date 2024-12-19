import rasterio
import os
from utils import reproject_gpdf , calculate_zonal_stats  , reproject_and_clip_raster, raster_sum_mean
import pandas as pd
import geopandas as gpd
from pathlib import Path

def export_urbanland(shp, vector_file, stat, section, data ,tables, rasters, country, city):
    for subdir in section:
        combined_results = []
        extension = '.tif'
        crs = 4326
        stats_to_get = stat[0] #list_statistics(stat)
        for root, dirs_list, files_list in os.walk( os.path.join(data, subdir)):
            for file_name in files_list:
                if os.path.splitext(file_name)[-1] == extension:
                    file_name_path = os.path.join(root, file_name)
                    file_name = os.path.splitext(file_name)[0]
                    print(file_name,  file_name_path, subdir)
                    Year = file_name[5:9].upper() 
                    ssp = file_name[0:4].upper() 
                    # combined_results.append(df)
                    out_rst = f'{rasters}/processed_{subdir}_{file_name}.tif'
                    #clip and export rasters for the pertinent ssp and year combo
                    rescaling_factor=1
                    reproject_and_clip_raster(input_raster=file_name_path, shapefile=shp, output_raster=out_rst, target_epsg=crs, multiply_factor=rescaling_factor)
                    
                    # Zonals
                    raster_path_clipped = str(Path(out_rst))
                    total_sum, mean_value = raster_sum_mean(raster_path_clipped)
                    print(f"Sum: {total_sum}, Mean: {mean_value}")
                    # Create a dictionary for the current row
                    row_data = { 'Scenario': ssp, 'Year': Year, f'{stats_to_get}': mean_value}
                    # Convert dictionary to DataFrame and append it to all_data
                    row_df = pd.DataFrame([row_data])  # Make sure to wrap row_data in a list to create a single-row DataFrame
                    # all_data = pd.concat([all_data, row_df], ignore_index=True)  # Use ignore_index to reindex after each append
                    combined_results.append(row_df)
                    print(f"row_df: {row_df}")

        # Append vertically and then pivot horizontally.
        df = gpd.GeoDataFrame( pd.concat( combined_results, ignore_index=True) )
        # Wide
        df_tranformed= df.pivot(index=['Scenario'], columns='Year', values=stats_to_get)
        # Long
        df_graph = df_tranformed.melt(ignore_index=False).reset_index()
        # export csv of wide
        df_tranformed.to_csv(f"{tables}/{country}_{city}_{subdir}.csv")
        print(df_tranformed)
 