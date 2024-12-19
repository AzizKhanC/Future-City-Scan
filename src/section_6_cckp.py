import xarray as xr
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

def export_cckp_data( vector_file, section, data ,tables, country, city):
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
                    ssp = file_name.split("-")[-3].upper() 

                    # ssp = subdir.upper()  #Already created 
                    # print(file_name,  file_name_path, subdir, Year, ssp, variable_name)
                    # dataset = xr.open_dataset(file_name_path, decode_coords="all")
                    dataset = xr.open_dataset(file_name_path)
                    clean_var_name= variable_name.replace('-', '_')
                    clean_var_name= clean_var_name.replace(':', '')
                    dataset=dataset.rename(name_dict={f'{variable_name}':f'{clean_var_name}'})
                    var_long_label= dataset.variables[clean_var_name].attrs['long_name'].capitalize()
                    var_long_label= var_long_label.replace(':', '')
                    # print(variable_name ,"----", clean_var_name, "----", var_long_label)
                    # start_date = "2040-01-01"
                    # end_date = "2040-12-31"
                    gpdf = vector_file # gpd.read_file(shp) #
                    gpdf['centroid'] = gpdf['geometry'].centroid
                    gpdf['lat'] = gpdf['centroid'].y
                    gpdf['lon'] = gpdf['centroid'].x
                    lat= gpdf['centroid'].y
                    lon= gpdf['centroid'].x
                    # df = dataset[clean_var_name].sel(time=slice(start_date, end_date), lat=slice(int(lat) , int(lat)),lon=slice(int(lon) , int(lon)))
                    df = dataset[clean_var_name].sel( lat=slice(int(lat) , int(lat)),lon=slice(int(lon) , int(lon)))
                    df=df.to_dataframe( )
                    df=df.dropna().reset_index()
                    df[subdir]=df[clean_var_name]
                    df["Variable"]=var_long_label 
                    df["SSP"]=ssp
                    try:
                        df_date = df[df[[subdir]].apply(pd.to_datetime, errors='coerce').isna().all(axis=1)]
                        df_date[subdir] = df_date[subdir] / pd.to_timedelta(1, unit='D')
                        appended_data.append(df_date)
                    except:
                        pass
                        # print(f"Passing {var_long_label}")
                        appended_data.append(df)
        appended_data = pd.concat(appended_data, axis=0, join='inner').sort_index()
        # appended_data = pd.concat(appended_data, axis=0).sort_index()
        appended_data.to_csv(f"{tables}/{country}_{city}_{subdir}_panel.csv")
    return appended_data

def export_cckp_graph(df , x_var_name, y_var_names, section, maps, country, city):
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
            ax = sns.lineplot(data=sub_df, x=x_var_name, 
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

    return df #print(f"{subdir} graph exported {df}")
