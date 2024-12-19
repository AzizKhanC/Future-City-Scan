from utils import set_paths
import yaml
import geopandas as gpd
import section_1_population 

# if __name__ == "main" :
# load city inputs files, set paths
with open("city_inputs.yml", 'r') as f:
    city_inputs = yaml.safe_load(f)
print(city_inputs['base_dir'])
base_dir=city_inputs['base_dir']
country=city_inputs['country_name']
city=city_inputs['city_name']

vector_file = gpd.read_file(city_inputs['AOI_path']).to_crs(epsg = 4326)
features = vector_file.geometry
aoi_bounds = vector_file.bounds
data, shapefiles , maps , rasters , output ,tables = set_paths(base_dir)

section=["popdynamics"]
stat=["sum"]
section_1_population.export_popdynamics(vector_file, stat, section, data ,tables, rasters, country, city)