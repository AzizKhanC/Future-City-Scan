import os  #, geojson
from pathlib import Path
import geopandas as gpd
import warnings
from utils import set_paths
import yaml
warnings.filterwarnings('ignore')
import pandas as pd


# socio_econmic=False
cckp=True
socio_econmic=True
WGS84=4326
EPSG_str= 'EPSG:4326'


# load city inputs files, set paths
with open("configs.yml", 'r') as f:
    configs = yaml.safe_load(f)
print(configs['base_dir'])
base_dir=configs['base_dir']
country=configs['country_name']
city=configs['city_name']

shp=configs['AOI_path']
AOI_path_countries=configs['AOI_path_countries']
vector_file = gpd.read_file(configs['AOI_path']).to_crs(epsg = 4326)
features = vector_file.geometry
aoi_bounds = vector_file.bounds
aoi_file=vector_file
data, shapefiles , maps , rasters , output ,tables = set_paths(base_dir)

all_countries = gpd.read_file((Path(shapefiles)/"WB_countries_Admin0_10m.shp"))
country_shp = all_countries[all_countries.NAME_EN == country]
country_shp.to_file(f"{Path(shapefiles)}/{country}.shp")
country_shp=f"{Path(shapefiles)}/{country}.shp"




# # if '__name__' == 'main':
from src import section_1_population ,section_1_1_population,  section_2_gdp, section_2_1_gdp, section_3_urbanland, section_4_heatflux, \
    section_5_urbanheatisland, section_6_cckp, section_7_extremeheat, section_8_infra_demographic_flood_exposure, \
        section_9_demographics, section_10_cyclones, section_11_erosion , section_12_post_processing


country_iso3 = 'BGD'
section= ['demographic']
country_name="Bangladesh"

section_1_population.create_updated_ssp3_dataset(country_iso3,country_name, city, section, data, tables )
    
section=["popdynamics"]
stat=["sum"]
section_1_population.export_popdynamics(shp, vector_file, stat, section, data ,tables, rasters, country, city)

section= ['demographic']
section_2_gdp.create_updated_ssp3_dataset(country_iso3,country_name, city, section, data, tables )
    
section= ['gdp'] 
section_2_gdp.export_gdp(shp, vector_file, stat, section, data ,tables, rasters, country, city)
section= ['urbanland']  
stat= ['mean']
section_3_urbanland.export_urbanland(shp, vector_file, stat, section, data ,tables, rasters, country, city)
section= ['heatflux']
section_4_heatflux.export_heatflux(shp, vector_file, stat, section, data ,tables, rasters, country, city)

# Clipping excludes touching all true retctify this
section= ['urbanheatisland']
stat= ['mean']
section_5_urbanheatisland.export_urbanheatisland(shp, vector_file, stat, section, data ,tables, rasters, country, city)

section= ['Precipitation']
appended_data= section_6_cckp.export_cckp_data( vector_file, section, data ,tables, country, city)
export_graph= section_6_cckp.export_cckp_graph(df=appended_data , x_var_name='time', y_var_names='Variable', section=section, maps=maps, country=country, city=city)

section= ['Temperature']
appended_data= section_6_cckp.export_cckp_data( vector_file, section, data ,tables, country, city)
export_graph= section_6_cckp.export_cckp_graph(df=appended_data , x_var_name='time', y_var_names='Variable', section=section, maps=maps, country=country, city=city)

# section= ['Extreme Heat Days']
# df= section_7_extremeheat.export_extreme_heat_cckp_data(vector_file, section, data , rasters, tables, country, city)
# export_graph= section_7_extremeheat.export_extreme_heat_cckp_graph(df=df , x_var_name='time', y_var_names='Variable',maps=maps, section=section,country=country, city=city)   


flood_bins = [15,30,50,100] 
revised_bins_list = [15,30,50,100] 
numberCategories = len(flood_bins)
rps = configs['flood']['rps']
flood_threshold = configs['flood']['threshold']
flood_years = configs['flood']['year']
flood_ssps = configs['flood']['ssp']

section_8_infra_demographic_flood_exposure.preprocess_fathom(output, city, aoi_file, country)
df2=section_8_infra_demographic_flood_exposure.conduct_pop_exposure(output,tables,rasters, vector_file, revised_bins_list , flood_bins, numberCategories, city, country)
cmap='hsv'
section_8_infra_demographic_flood_exposure.create_pop_exposure_barchart(df2, revised_bins_list,cmap, maps)

section_8_infra_demographic_flood_exposure.conduct_infra_exposure_analysis(output, tables ,shapefiles, flood_bins,flood_years, rps ,flood_ssps,  city, country)
cmap='turbo'
section_8_infra_demographic_flood_exposure.export_infra_exposure_barcharts(tables, revised_bins_list, cmap, maps,city, country)

base_path = output
directories_to_remove = ["Coastal", "Pluvial", "Fluvial"]
section_8_infra_demographic_flood_exposure.post_process_dir_cleaning(base_path, directories_to_remove)




country_iso3 = 'BGD'
section= ['demographic']
city_pop_ssp2, country_pop_ssp2 = section_9_demographics.demographic_projection(city=city, 
                                                        country_iso3=country_iso3, 
                                                        country=country, 
                                                        ssp_to_use='SSP2', section=section, data=data)

city_pop_ssp5, country_pop_ssp5 = section_9_demographics.demographic_projection(city=city, 
                                                        country_iso3=country_iso3, 
                                                        country=country, 
                                                        ssp_to_use='SSP5', section=section, data=data)
city_pop = city_pop_ssp2._append(city_pop_ssp5)
city_pop = city_pop.loc[city_pop['Year']<=2070]
city_pop.reset_index(inplace=True, drop=True)

country_pop = country_pop_ssp2._append(country_pop_ssp5)
country_pop = country_pop.loc[country_pop['Year']<=2070]
country_pop.reset_index(inplace=True, drop=True)
year1_list=[2020, 2050]
year2_list=[2050, 2070]
ssp_list=["SSP2", "SSP5"]
for ssp in ssp_list:
    section_9_demographics.pop_lineplot(city_pop, ssp, section, maps, city, country)
    for year1, year2 in zip(year1_list, year2_list):
        print(year1, year2, ssp)
        export_pyramid= section_9_demographics.pyramid(city_pop, year1, year2, ssp, section, maps, city, country)


section= ['tropicalcyclones']
gdf =vector_file  #gpd.read_file(shp).to_crs(WGS84) #r"inputs/Ukhiya.shp"
output_cyclone = section_10_cyclones.get_wind_speed(tables,data, gdf,section, country, city)
section_10_cyclones.visualize_results(maps,section, output_cyclone, country, city)

section= ['globalerosion']
df=section_11_erosion.process_erosion(tables,data,section, shp, country, city)
section_11_erosion.map_storms(appended_data=df, shp=shp)
gdf_shoreline_points=section_11_erosion.create_points(country, shapefiles, df)

years_multiplier='no_temporal_cumulative'
transect_gdf, gdf_sorted=section_11_erosion.create_shorelines_transect(country, shapefiles, df, years_multiplier)
# transect_gdf.explore()
transect_gdf_attributes=section_11_erosion.trasfer_attributes_to_transects(country, shapefiles, transect_gdf, gdf_sorted,years_multiplier)

section=["popdynamics"]
stat=["sum"]

section_1_1_population.export_popdynamics(shp=country_shp, vector_file=vector_file, stat=stat, section=section, data=data ,tables=tables, rasters=rasters, country=country, city=city)
# Example usage
folder_path = rasters
cleaned_section = "".join(map(str, section))
output_csv_path = f"{tables}/{cleaned_section}_city_to_country_ratios.csv"
section_1_1_population.process_rasters_by_prefix_suffix(folder_path, output_csv_path)

section= ['gdp'] 
stat=["sum"]
section_2_1_gdp.export_gdp(shp=country_shp, vector_file=vector_file, stat=stat, section=section, data=data ,tables=tables, rasters=rasters, country=country, city=city)
# Example usage
folder_path = rasters
cleaned_section = "".join(map(str, section))
output_csv_path = f"{tables}/{cleaned_section}_city_to_country_ratios.csv"
section_2_1_gdp.process_gdp_rasters(folder_path, output_csv_path)
print(f"Results exported to {output_csv_path}")


