import rasterio
import os
from utils import reproject_gpdf , calculate_zonal_stats  , reproject_and_clip_raster, raster_sum_mean
import pandas as pd
import geopandas as gpd
from pathlib import Path
from osgeo import gdal
import numpy as np
import csv

def export_gdp(shp, vector_file, stat, section, data ,tables, rasters, country, city):
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
                    # Year = file_name.split('_')[1].upper() 
                    # ssp = file_name.split('_')[0].upper() 
                    file_name = os.path.splitext(file_name)[0]
                    Year = file_name[3:7].upper() 
                    ssp = file_name[8:14].upper()                     
                    if int(Year)>2020:
                        try:
                            # var= 'IIASA-WiC POP 2023' # 'IIASA GDP 2023', 
                            var='IIASA GDP 2023'
                            input_csv=f"{tables}/{country}_{city}_{var}_rescaled.csv"
                            year=Year #2025
                            ssp_value= ssp # "SSP1"
                            rescaling_factor=get_rescaling_factor(input_csv, year, ssp_value)
                            out_rst = f'{rasters}/{country}_processed_{subdir}_{file_name}.tif'
                            print(f"rescaling_factor {rescaling_factor}")
                            reproject_and_clip_raster(input_raster=file_name_path, shapefile=shp, output_raster=out_rst, target_epsg=crs, multiply_factor=rescaling_factor)
                            print(f"clip_and_export_rescale_raster done 1")
                            
                            # Zonals
                            raster_path_clipped = str(Path(out_rst))
                            total_sum, mean_value = raster_sum_mean(raster_path_clipped)
                            print(f"Sum: {total_sum}, Mean: {mean_value}")
                            # Create a dictionary for the current row
                            row_data = { 'Scenario': ssp, 'Year': Year, f'{stats_to_get}': int(total_sum)}
                            # Convert dictionary to DataFrame and append it to all_data
                            row_df = pd.DataFrame([row_data])  # Make sure to wrap row_data in a list to create a single-row DataFrame
                            # all_data = pd.concat([all_data, row_df], ignore_index=True)  # Use ignore_index to reindex after each append
                            combined_results.append(row_df)
                            print(f"row_df: {row_df}")
                            
                        except:
                            print(f"Skipped year {Year}, ssp_value {ssp}")
                    elif int(Year)==2020:
                        try:
                            
                            out_rst = f'{rasters}/{country}_processed_{subdir}_{file_name}.tif'
                            rescaling_factor=1
                            reproject_and_clip_raster(input_raster=file_name_path, shapefile=shp, output_raster=out_rst, target_epsg=crs, multiply_factor=rescaling_factor)
                            print(f"clip_and_export_rescale_raster done 2")
                                                        
                            # Zonals
                            raster_path_clipped = str(Path(out_rst))
                            total_sum, mean_value = raster_sum_mean(raster_path_clipped)
                            print(f"Sum: {total_sum}, Mean: {mean_value}")         
                            # Create a dictionary for the current row
                            row_data = { 'Scenario': ssp, 'Year': Year, f'{stats_to_get}': int(total_sum)}
                            # Convert dictionary to DataFrame and append it to all_data
                            row_df = pd.DataFrame([row_data])  # Make sure to wrap row_data in a list to create a single-row DataFrame
                            # all_data = pd.concat([all_data, row_df], ignore_index=True)  # Use ignore_index to reindex after each append
                            combined_results.append(row_df)
                            print(f"row_df: {row_df}")
                            
                        except:
                            print(f"Skipped year {Year}, ssp_value {ssp}")

    # Append vertically and then pivot horizontally.
    df = gpd.GeoDataFrame( pd.concat( combined_results, ignore_index=True) )
    # Wide
    df_tranformed= df.pivot(index=['Scenario'], columns='Year', values=stats_to_get)
    # export csv of wide
    df_tranformed.to_csv(f"{tables}/country_level_{country}_{subdir}.csv")
    
    
    
def get_rescaling_factor(input_csv, year, ssp_value):
    df= pd.read_csv(input_csv)
    df=df[[f"rescaling_factor_{year}_2023_div_{year}_2013", "SCENARIO"]] 
    rescaling_factor=df[df['SCENARIO']==ssp_value][f"rescaling_factor_{year}_2023_div_{year}_2013"].item()
    print(f"year--> {year}  ....ssp_value-->> {ssp_value}...rescaling_factor--->> {rescaling_factor}")
    return rescaling_factor



def get_ssp2_2013_dataset(data,city, country_iso3, section, dataset,var):
    gc_city_folder = os.path.join(data, f'{section[0]}/GC_countries/')
    ssp_master_fn = os.path.join(data, f'{section[0]}/{dataset}')
    # Step 0.1: Get country-level projections
    if ".csv" in dataset:
        ssp_master = pd.read_csv(ssp_master_fn)
    print(ssp_master.columns)
    # ssp_master[ssp_master.columns] = ssp_master[ssp_master.columns].astype(str).apply(lambda col: col.str.upper())
    ssp_master.columns = [col.upper() for col in ssp_master if col in ssp_master.columns]
    # ssp_master['REGION'] = ssp_master['REGION'].str[:4]
    ssp_master['SCENARIO'] = ssp_master['SCENARIO'].str[:4]
    print("capital---", ssp_master.columns)
    ssp_country = ssp_master.loc[ssp_master['REGION']==country_iso3]
    print(f"variables----->>>> {ssp_country['MODEL'].unique()}")

    ssp_country = ssp_country.loc[ssp_country['MODEL']==var]
    ssp_country.reset_index(inplace=True, drop=True)
    if len(ssp_country)==0:
        print('Country not found in SSP database. Check if ISO3 name of the country is correct, and whether the country is actually available in the SSP database.')
        return 0,0
    else:
        del ssp_master
    
    
    # Step 0.2: Get city's historical demographic change
    global_cities = pd.read_csv(gc_city_folder+'GC_{}.csv'.format(country_iso3))
    gc_city = global_cities.loc[global_cities['Location']==city]
    if len(gc_city)==0:
        print('City not found in the Global Cities database. Manually check if the city name spelling is correct, and whether the city is actually available in the Global Cities database.')
        return 0,0
    else:
        del global_cities
    iso3=gc_city['iso3'].unique()[0]
    country_name=gc_city['Country'].unique()[0]
    print(iso3, country_name.title())
    ssp_country['iso3']=iso3.title()
    ssp_country['REGION']=country_name.title()
    
    return ssp_country, country_name, country_iso3

# 2023 ssp3
def get_ssp3_2023_dataset(data, country, section, dataset,var):
    ssp_master_fn = os.path.join(data, f'{section[0]}/{dataset}')
    # Step 0.1: Get country-level projections
    if ".csv" in dataset:
        ssp_master = pd.read_csv(ssp_master_fn)
    print(ssp_master.columns)
    ssp_master.columns = [col.upper() for col in ssp_master if col in ssp_master.columns]
    print("capital---", ssp_master.columns)
    ssp_country = ssp_master.loc[ssp_master['REGION']==country]
    ssp_country = ssp_country.loc[ssp_country['MODEL']==var]
    ssp_country.reset_index(inplace=True, drop=True)
    if len(ssp_country)==0:
        print('Country not found in SSP database. Check if ISO3 name of the country is correct, and whether the country is actually available in the SSP database.')
        return 0,0
    else:
        del ssp_master
    return ssp_country 


def create_updated_ssp3_dataset(country_iso3,country_name, city, section, data, tables ):
    country_iso3 = 'BGD'
    var= 'IIASA GDP' #'IIASA-WiC POP' # 'IIASA GDP 2023', 
    var_proper="GDP|PPP" #"Population"
    dataset=r"SspDb_country_data_2013-06-12.csv"
    ssp_country_2013, country_name, country_iso3=get_ssp2_2013_dataset(data,city, country_iso3, section, dataset,var)
    ssp_country_2013=ssp_country_2013[ssp_country_2013['VARIABLE']==var_proper]
    ssp_country_2013
    # # Get gdp defkator and inflate the gdp to 2017 USD 
    dataset=r'gdp_deflator.csv'
    deflator= os.path.join(data, f'{section[0]}/{dataset}')
    deflator = pd.read_csv(deflator)
    deflator=deflator[deflator["Country Code"]==country_iso3]
    deflator_multiplier=(deflator['2017']/deflator['2005']).item()
    print(deflator_multiplier)

    # Multiply all columns by the defaltor multiplier
    cols = ssp_country_2013.columns.tolist()
    # for j, s in enumerate(cols):  
    for i in range(2025,2155,5):
        try: 
            # a = int(s) 
            y1=f"{i}"
            print(f"y1={y1}, i={i}")
            ssp_country_2013[f"{i}"] = ssp_country_2013[y1]*deflator_multiplier
        except ValueError: 
                    print(f'{s} was not an integer') 
    var= 'IIASA GDP 2023'
    section= ['demographic']
    dataset=r"1706548837040-ssp_basic_drivers_release_3.0_full.csv"
    ssp_country=get_ssp3_2023_dataset(data=data, country=country_name, section=section, dataset=dataset, var=var) 

    df_merged=ssp_country_2013.merge(ssp_country, how='outer', 
                                    on=['SCENARIO','REGION','VARIABLE'], 
                                    suffixes=('_2013', '_2023'),
                                    indicator= True)

    cols = df_merged.columns.tolist()
    for j, s in enumerate(cols):  
        for i in range(2025,2105,5):
            try: 
                a = int(s) 
                y1=f"{i}_2013"
                y2=f"{i}_2023"
                # print(f"y1={y1}, y2={y2}, s={s}")
                df_merged[f"rescaling_factor_{y2}_div_{y1}"] = df_merged[y2]/df_merged[y1]
                df_merged[f"updated_{i}"] = df_merged[y1]*df_merged[f"rescaling_factor_{y2}_div_{y1}"]
            except ValueError: 
                        print(f'{s} was not an integer')                                 
    df_merged.to_csv(f"{tables}/{country_name}_{city}_{var}_rescaled.csv")

    return df_merged

import os
import csv
from osgeo import gdal
import numpy as np

def gdp_ratio(raster1_path, raster2_path):
    """
    Calculate the ratio of the total values of two GDP rasters.

    Parameters:
        raster1_path (str): Path to the first raster file.
        raster2_path (str): Path to the second raster file.

    Returns:
        float: Ratio of raster1's total sum to raster2's total sum.
    """
    def read_raster_as_array(raster_path):
        dataset = gdal.Open(raster_path)
        if dataset is None:
            raise FileNotFoundError(f"Could not open {raster_path}")
        band = dataset.GetRasterBand(1)
        array = band.ReadAsArray()
        dataset = None
        return array

    raster1_array = read_raster_as_array(raster1_path)
    raster2_array = read_raster_as_array(raster2_path)
    
    raster1_sum = np.nansum(raster1_array)
    raster2_sum = np.nansum(raster2_array)
    
    if raster2_sum == 0:
        raise ValueError("Sum of raster2 values is zero, division by zero is not allowed.")
    
    return raster1_sum / raster2_sum

def process_gdp_rasters(folder_path, output_csv_path):
    """
    Process GDP rasters with specific prefixes and a common wildcard suffix, 
    compute ratios, and export results to a CSV.

    Parameters:
        folder_path (str): Path to the folder containing rasters.
        output_csv_path (str): Path to the output CSV file.

    Returns:
        None
    """
    # List all raster files in the folder
    raster_files = [f for f in os.listdir(folder_path) if f.endswith(('.tif', '.TIF'))]
    
    # Dictionaries to hold rasters by their suffix
    processed_files = {}
    bangladesh_files = {}

    # Categorize rasters by their prefix and suffix
    for file in raster_files:
        if file.startswith("processed_"):
            suffix = file[len("processed_"):]  # Extract the suffix after the prefix
            processed_files[suffix] = os.path.join(folder_path, file)
        elif file.startswith("Bangladesh_processed_"):
            suffix = file[len("Bangladesh_processed_"):]  # Extract the suffix after the prefix
            bangladesh_files[suffix] = os.path.join(folder_path, file)
    
    # Open a CSV file to write results
    with open(output_csv_path, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header
        csv_writer.writerow(["Raster 1", "Raster 2", "GDP Ratio"])
        
        # Compute ratios for matching suffixes
        for suffix, raster1_path in processed_files.items():
            if suffix in bangladesh_files:
                raster2_path = bangladesh_files[suffix]
                try:
                    ratio = gdp_ratio(raster1_path, raster2_path)
                    csv_writer.writerow([os.path.basename(raster1_path), os.path.basename(raster2_path), ratio])
                except Exception as e:
                    print(f"Error processing {raster1_path} and {raster2_path}: {e}")
            else:
                print(f"No matching raster found for suffix '{suffix}' in Bangladesh files.")
