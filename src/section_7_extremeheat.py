import xarray as xr
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from osgeo import gdal, osr

def export_extreme_heat_cckp_data(vector_file, section, data , rasters, tables, country, city):
    appended_data = []
    for subdir in section:
        extension = '.nc'
        for root, dirs_list, files_list in os.walk( os.path.join(data, subdir)):
            for file_name in files_list:
                if os.path.splitext(file_name)[-1] == extension:
                    file_name_path = os.path.join(root, file_name)
                    file_name = os.path.splitext(file_name)[0]
                    # print(file_name,  file_name_path, subdir)
                    variable_name = file_name.split("_")[0] #.upper()
                    Year = file_name.split("_")[-1].upper()
                    ssp = file_name.split("-")[-2].upper() 
                    ssp = ssp.split("_")[0].upper() 
                    # ssp = subdir.upper()  #Already created 
                    # print(file_name,  file_name_path, subdir, Year, ssp, variable_name)
                    # dataset = xr.open_dataset(file_name_path, decode_coords="all")
                    dataset = xr.open_dataset(file_name_path)
                    print(f"Varname-->>{list(dataset.keys())}")

                    clean_var_name= variable_name.replace('-', '_')
                    clean_var_name= clean_var_name.replace(':', '')
                    dataset=dataset.rename(name_dict={f'{variable_name}':f'{clean_var_name}'})
                    var_long_label= dataset.variables[clean_var_name].attrs['long_name'].capitalize()
                    var_long_label= var_long_label.replace(':', '')
                    print(variable_name ,"----", clean_var_name, "----", var_long_label)
                    # start_date = "2040-01-01"
                    # end_date = "2040-12-31"
                    gpdf = vector_file # gpd.read_file(shp) #
                    gpdf['centroid'] = gpdf['geometry'].centroid
                    gpdf['lat'] = gpdf['centroid'].y
                    gpdf['lon'] = gpdf['centroid'].x
                    # lat= gpdf['centroid'].y
                    # lon= gpdf['centroid'].x

                    lat= gpdf['centroid'].y[0]
                    lon= gpdf['centroid'].x[0]
                    # Get the bounds for the entire GeoDataFrame
                    bounds = vector_file.total_bounds
                    # The total_bounds attribute returns a tuple in the format:
                    min_long, min_lat, max_long, max_lat = bounds
                    # Print the results
                    print(f"Min Latitude: {min_lat}--Max Latitude: {max_lat}-- Min Longitude: {min_long}--- Max Longitude: {max_long}")
                    print(f"Centroid Latitude: {lat}--  Centroid lon: {lon}")
                
                    df = dataset[clean_var_name].sel( lat=slice(int(lat) , int(lat)),lon=slice(int(lon) , int(lon)))
                    df=df.to_dataframe( )
                    df=df.dropna().reset_index()
                    print(f"values for the variable: {df[clean_var_name].value_counts()}")
                    df[subdir]=df[clean_var_name]
                    df["Variable"]=var_long_label 
                    df["SSP"]=ssp
                    try:
                        df_date = df[df[[subdir]].apply(pd.to_datetime, errors='coerce').isna().all(axis=1)]
                        df_date[subdir] = df_date[subdir] / pd.to_timedelta(1, unit='D')
                        # appended_data.append(df_date)
                        print(f"Max value of days: {df_date[subdir].max()}---Mean value of days: {df_date[subdir].mean()}")
                        max_value=df_date[subdir].max()
                    except:
                        pass
                        print(f"Passing {var_long_label}")
                        # appended_data.append(df)

                    if max_value<0.00001:
                        # offset=(1/111)*50
                        offset=0.25
                        print(f"setting {offset}")
                        print(f"int(lat-offset) , int(lat+offset): {(lat-offset)} , {(lat+offset)}---Mint(lon-offset) , int(lon+offset)): {(lon-offset)} , {(lon+offset)}")
                        df = dataset[clean_var_name].sel( lat=slice((lat-offset) , (lat+offset)),lon=slice((lon-offset) , (lon+offset)))
                        df=df.to_dataframe( )
                        df=df.dropna().reset_index()
                        print(f"values for the variable: {df[clean_var_name].value_counts()}-- max value: {df[clean_var_name].max()}")
                        df[subdir]=df[clean_var_name]
                        df["Variable"]=var_long_label 
                        df["SSP"]=ssp
                        try:
                            df_date = df[df[[subdir]].apply(pd.to_datetime, errors='coerce').isna().all(axis=1)]
                            df_date[subdir] = df_date[subdir] / pd.to_timedelta(1, unit='D')
                            appended_data.append(df_date)
                            print(f"Max value of days: {df_date[subdir].max()}---Mean value of days: {df_date[subdir].mean()}")
                        except:
                            pass
                            # print(f"Passing {var_long_label}")
                            appended_data.append(df)
                    else:
                        offset >= 0.00001
                        print(f"Not setting setting {offset}")
                        df = dataset[clean_var_name].sel( lat=slice(lat , lat),lon=slice(lon , lon))
                        df=df.to_dataframe( )
                        df=df.dropna().reset_index()
                        print(f"values for the variable: {df[clean_var_name].value_counts()}-- max value: {df[clean_var_name].max()}")
                        df[subdir]=df[clean_var_name]
                        df["Variable"]=var_long_label 
                        df["SSP"]=ssp
                        try:
                            df_date = df[df[[subdir]].apply(pd.to_datetime, errors='coerce').isna().all(axis=1)]
                            df_date[subdir] = df_date[subdir] / pd.to_timedelta(1, unit='D')
                            appended_data.append(df_date)
                            print(f"Max value of days: {df_date[subdir].max()}---Mean value of days: {df_date[subdir].mean()}")
                        except:
                            pass
                            # print(f"Passing {var_long_label}")
                            appended_data.append(df)

                    print(f"step 2")
                    # Load the NetCDF file
                    data_1 = xr.open_dataset(file_name_path)
                    ###############
                    offset=1
                    print(f"setting {offset}")
                    print(f"int(lat-offset) , int(lat+offset): {(lat-offset)} , {(lat+offset)}---Mint(lon-offset) , int(lon+offset)): {(lon-offset)} , {(lon+offset)}")
                    # data_1 = data_1[variable_name].sel( lat=slice((lat-offset) , (lat+offset)),lon=slice((lon-offset) , (lon+offset)))
                    print(f"step 3")
                    print(f"data_1 {data_1}")
                    ###############
                    # Select the variable of interest
                    # var_data = data_1[variable_name]
                    var_data = data_1[variable_name].sel( lat=slice((lat-offset) , (lat+offset)),lon=slice((lon-offset) , (lon+offset)))

                    # Mask the _FillValue (e.g., 1e+20) and set them to NaN
                    if "_FillValue" in var_data.attrs:
                        fill_value = var_data.attrs["_FillValue"]
                        var_data = var_data.where(var_data != fill_value, np.nan)
                    
                    # Fill NaN values with 0 to handle missing data
                    var_data = var_data.fillna(0)
                    
                    print(f"step 4")
                    # Group data by month and calculate the monthly sum
                    monthly_sum = var_data.groupby('time.month').sum(dim='time')
                    
                    # Extract longitude and latitude
                    lons = var_data['lon'].values
                    lats = var_data['lat'].values
                    
                    # Ensure lons are increasing
                    if np.any(np.diff(lons) < 0):
                        lons = lons[::-1]
                        monthly_sum = monthly_sum[:, :, ::-1]

                    # Ensure lats are in the correct order
                    if np.any(np.diff(lats) < 0):
                        lats = lats[::-1]
                        monthly_sum = monthly_sum[:, ::-1, :]
                    print(f"step 5")
                    # Define pixel resolution
                    pixel_width = (lons.max() - lons.min()) / (len(lons) - 1)
                    pixel_height = (lats.max() - lats.min()) / (len(lats) - 1)  # Ensure height is positive

                    # Loop over each month to save the monthly sum as a GeoTIFF
                    for month in range(1, 13):
                        month_data = monthly_sum.sel(month=month).values
                        
                        # Handle NaN values
                        month_data = np.nan_to_num(month_data, nan=0.0)

                        # Create GeoTIFF file name
                        output_tiff = f"{rasters}/{file_name}_monthly_sum_{month:02d}_new.tif"

                        # Create a new GeoTIFF file
                        driver = gdal.GetDriverByName('GTiff')
                        out_raster = driver.Create(
                            output_tiff, 
                            month_data.shape[1], 
                            month_data.shape[0], 
                            1, 
                            gdal.GDT_Float32
                        )
                        
                        # Define the geotransformation
                        # Ensure that the Y-coordinates (lats) are set correctly for upper-left corner
                        geotransform = (lons.min(), pixel_width, 0, lats.max(), 0, -pixel_height)
                        out_raster.SetGeoTransform(geotransform)
                        
                        # Set projection (assuming EPSG:4326)
                        srs = osr.SpatialReference()
                        srs.ImportFromEPSG(4326)
                        out_raster.SetProjection(srs.ExportToWkt())
                        
                        # # Write the data to the first band, flipping it vertically
                        out_raster.FlushCache()
                        out_raster = None
                        print(f"GeoTIFF saved as {output_tiff}")

        appended_data = pd.concat(appended_data, axis=0, join='inner').sort_index()
        appended_data.to_csv(f"{tables}/{country}_{city}_{subdir}_panel.csv")
    return appended_data

def export_extreme_heat_cckp_graph(df , x_var_name, y_var_names, maps, section, country, city):
    subdir=section[0]
    df["date"] = pd.to_datetime(df[f"{x_var_name}"]) #.month
    # df['month_year'] = df['date_column'].dt.to_period('M')
    # y_var_names=df[y_var_names].unique()
    for ssp in df["SSP"].unique():
        for y_var in df[y_var_names].unique():
            sub_df = df[(df["SSP"]==ssp) & (df[y_var_names]==y_var)]
            sub_df=sub_df.sort_values([y_var_names, "SSP"], ascending=[True, True])
            # Set a Style
            sns.set_style('whitegrid')
            # Create plot
            ax = sns.lineplot(data=sub_df, x="date", 
                            y=subdir , errorbar=None, 
                            hue=y_var_names, style=y_var_names, markers=True
                            )
            # Set Title and Labels
            ax.set_title(f'{subdir} in {city} for {ssp}', 
            fontdict={'size': 13, 'weight': 'bold'})
            ax.set_xlabel(f'Month')
            # ax.set_ylabel(f'{y_var}')
            # Adjust the Legend
            plt.legend(frameon=False, loc='lower left', bbox_to_anchor=(0, -.3), ncol=1)
            # Format the date axis to be prettier.
            # ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
            # ax.xaxis.set_minor_locator(mdates.DayLocator())
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(interval_multiples=False))
            # # Increase size and change color of axes ticks
            GREY91 = "#e8e8e8"
            ax.tick_params(axis="x", length=12, color=GREY91)
            ax.tick_params(axis="y", length=8, color=GREY91)
            # Remove the Spines
            sns.despine()
            plt.show()
            ax.figure.savefig(f"{maps}/{country}_{city}_{subdir}_graph.png",  bbox_inches='tight', dpi=300) #, transparent=True  

    return sub_df 