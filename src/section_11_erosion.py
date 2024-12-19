# Incorporate the tropical cyclones 
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shapefile
import geopandas as gpd
import contextily as ctx
from shapely.geometry import shape, Point 
import glob
from shapely.geometry import Point, LineString
from scipy.spatial import distance_matrix
import math

def process_erosion(tables,data,section, shp, country, city):
    # section
    polygon=get_city_poolygon(shp)
    column_names = ['Latitude','Longitude' , 'percentile_1', 'percentile_5', 'percentile_17', 'percentile_50','percentile_83', 'percentile_95', 'percentile_99']
    for subdir in section:      
        file_list = glob.glob(os.path.join( os.path.join(data, subdir), "globalErosionProjections_Long_Term_Change*.csv"))
        files = []
        appended_data=[]
        for file_path in file_list:
            file_name= os.path.basename(file_path)
            file_name = os.path.splitext(file_name)[0]
            year=file_name.split("_")[-1]
            rcp=file_name.split("_")[-2]
            
            # df_points= pd.read_csv(file_path)
            df_points= pd.read_csv(file_path,  names=column_names, header=None)
            # print("Read file:" , file_path)
            for lat,lon in zip(df_points.Latitude, df_points.Longitude):
                if check(lon, lat, polygon):
                    files.append(file_path)
                    # print(check(lon, lat))
                    sub_df = df_points[(df_points["Latitude"]==lat) & (df_points["Longitude"]==lon)]
                    sub_df["year"] = year
                    sub_df["rcp"] = rcp
                    appended_data.append(sub_df)
    appended_data = pd.concat(appended_data)
    # write DataFrame to an excel sheet 
    appended_data.to_csv(f"{tables}/{country}_{city}_{subdir}_erosion.csv")
    erosion_df=appended_data
    return erosion_df
    # return df_points
    
def get_city_poolygon(shp):
    # get the shapes
    r = shapefile.Reader(shp)
    shapes = r.shapes()
    # build a shapely polygon from your shape
    polygon = shape(shapes[0])   
    # read your shapefile
    return polygon


def map_storms(appended_data, shp):
    lon =appended_data["Longitude"]
    lat =appended_data["Latitude"]
    # lat = [28.6877899169922, 28.663863, 28.648287, 28.5429172515869]
    geometry = [Point(xy) for xy in zip(lon,lat)]
    wardlink = shp
    ward = gpd.read_file(wardlink, bbox=None, mask=None, rows=None)
    geo_df = gpd.GeoDataFrame(geometry = geometry)

    ward.crs = {'init':"epsg:4326"}
    geo_df.crs = {'init':"epsg:4326"}

    # plot the polygon
    ax = ward.plot(alpha=0.35, color='#d66058', zorder=1)
    # plot the boundary only (without fill), just uncomment
    #ax = gpd.GeoSeries(ward.to_crs(epsg=3857)['geometry'].unary_union).boundary.plot(ax=ax, alpha=0.5, color="#ed2518",zorder=2)
    ax = gpd.GeoSeries(ward['geometry'].unary_union).boundary.plot(ax=ax, alpha=0.5, color="#ed2518",zorder=2)

    # plot the marker
    ax = geo_df.plot(ax = ax, markersize = 20, color = 'red',\
                    marker = '*', zorder=3,
                    legend=True,  # Add legend 
                    legend_kwds={'loc':'upper right', 
                                    'bbox_to_anchor':(1, 1), 
                                                    'markerscale':1.01, 
                                                    'title_fontsize':'small', 
                                                    'fontsize':'x-small'
                                                    } ,
                                                    )
    ctx.add_basemap(ax, crs=geo_df.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)
    plt.xticks([])
    plt.yticks([])
        
    leg1 = ax.get_legend()
    # Set markers to square shape
    # for ea in leg1.legendHandles:
    #     ea.set_marker('s')
    # leg1.set_title(f'{legend_title}')
    # ax.title.set_text(f'{title}')
    # plt.tight_layout()
    # ax.figure.savefig(map_output)
    plt.show()


def check(lon, lat, polygon):
    # build a shapely point from your geopoint
    point = Point(lon, lat)
    # the contains function does exactly what you want
    return polygon.contains(point)


def create_points(country, shapefiles, df):

    # df = pd.read_csv(csv)
    print(f"Total Obs: {len(df)}")
    # df= df[df['country_id']==country]
    print(f"Subset Obs: {len(df)}")

    def create_point(row):
        if pd.notnull(row['Latitude']) and pd.notnull(row['Longitude']):
            return Point(row['Longitude'], row['Latitude'])
        return None  # Return None if lat/lon is missing

    df['geometry'] = df.apply(create_point, axis=1)
    # Drop rows with empty geometry
    df = df.dropna(subset=['geometry']).reset_index(drop=True)
    print(f"After dropping Nan-->>> Obs: {len(df)}")

    # Step 3: Convert the DataFrame to a GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    gdf.set_crs(epsg=4326, inplace=True)  # Set CRS to WGS84
    print(f"Last-->> Subset Obs: {len(gdf)}")
    gdf.to_file(f"{shapefiles}/{country}_shoreline_points.shp")
    return gdf


def create_shorelines_transect(country, shapefiles, df, years_multiplier):
    # df = pd.read_csv(csv)
    print(f"Total Obs: {len(df)}")
    # df= df[df['country_id']==country]
    print(f"Subset Obs: {len(df)}")
    df['geometry'] = df.apply(lambda row: Point(row['Longitude'], row['Latitude']) if pd.notnull(row['Latitude']) and pd.notnull(row['Longitude']) else None, axis=1)
    df = df.dropna(subset=['geometry']).reset_index(drop=True)
    # Convert to GeoDataFrame and set CRS to WGS84
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
    gdf_utm = gdf.to_crs(epsg=32632)  # Convert to UTM for distance calculation in meters
    
    # Step 2: Sort points by proximity using distance matrix
    coords = gdf_utm['geometry'].apply(lambda geom: (geom.x, geom.y)).tolist()
    dist_matrix = distance_matrix(coords, coords)

    visited_order = [0]  # Start with the first point
    current_index = 0

    while len(visited_order) < len(gdf_utm):
        distances = dist_matrix[current_index]
        distances[visited_order] = float('inf')  # Set visited points to infinity
        nearest_index = distances.argmin()
        visited_order.append(nearest_index)
        current_index = nearest_index

    # Reorder GeoDataFrame
    gdf_sorted = gdf_utm.iloc[visited_order].reset_index(drop=True)

    # Create LineString from the ordered points
    line = LineString(gdf_sorted.geometry.tolist())
    line_gdf = gpd.GeoDataFrame(geometry=[line], crs=gdf_utm.crs)

    # Ensure the GeoDataFrame has a CRS in meters (UTM for Tunisia: EPSG:32632)
    shoreline_gdf = gdf_sorted.to_crs(epsg=32632)

    # Convert the shoreline points to a single LineString
    # shoreline_line =  LineString(multi_line_gdf.geometry.tolist())
    shoreline_line =  LineString(shoreline_gdf.geometry.tolist())

    # Create an empty list to store one-sided perpendicular line segments (transects)
    transects = []
    cumulative= []
    # Iterate through each point in the LineString
    for i, point in enumerate(shoreline_line.coords):
        if i == len(shoreline_line.coords) - 1:
            break  # Skip the last point as there is no "next" point
        # Fetch the percentile_50 value for the corresponding point
        if years_multiplier=='temporal_cumulative':
            # Convert current point and next point to Point geometries
            start_point = Point(point)
            end_point = Point(shoreline_line.coords[i + 1])
            print(shoreline_gdf.head())
            percentile_50 = shoreline_gdf['percentile_50'].iloc[i]
            sign =1 if percentile_50 > 0 else -1
            Timespan=shoreline_gdf['percentile_50'].iloc[i]*shoreline_gdf['Timespan'].iloc[i] +  shoreline_gdf['intercept'].iloc[i]*sign

            transect_length = abs(Timespan)  # Use the absolute value for length
            # Determine direction: 90° if percentile_50 is positive, -90° if negative
            direction = -1 if percentile_50 >= 0 else 1
            cumulative.append(transect_length*direction)

            # Calculate segment angle of the shoreline segment between points
            dx = end_point.x - start_point.x
            dy = end_point.y - start_point.y
            segment_angle = math.atan2(dy, dx)
            # Calculate perpendicular angle with direction adjustment
            perp_angle = segment_angle + (math.pi / 2) * direction
            # Calculate transect endpoint coordinates (single-sided)
            transect_dx = math.cos(perp_angle) * transect_length
            transect_dy = math.sin(perp_angle) * transect_length
            # Create the transect starting from the current point and extending in one direction
            transect_end = Point(start_point.x + transect_dx, start_point.y + transect_dy)
            transect_line = LineString([start_point, transect_end])
            transects.append(transect_line)
        else:
            # Convert current point and next point to Point geometries
            start_point = Point(point)
            end_point = Point(shoreline_line.coords[i + 1])

            percentile_50 = shoreline_gdf['percentile_50'].iloc[i]
            Timespan=1
            # print(f"Timespan {Timespan}")
            transect_length = abs(percentile_50*int(Timespan))  # Use the absolute value for length
            # Determine direction: 90° if percentile_50 is positive, -90° if negative
            direction = -1 if percentile_50 >= 0 else 1
            cumulative.append(transect_length*direction)
            # Calculate segment angle of the shoreline segment between points
            dx = end_point.x - start_point.x
            dy = end_point.y - start_point.y
            segment_angle = math.atan2(dy, dx)
            # Calculate perpendicular angle with direction adjustment
            perp_angle = segment_angle + (math.pi / 2) * direction
            # Calculate transect endpoint coordinates (single-sided)
            transect_dx = math.cos(perp_angle) * transect_length
            transect_dy = math.sin(perp_angle) * transect_length
            # Create the transect starting from the current point and extending in one direction
            transect_end = Point(start_point.x + transect_dx, start_point.y + transect_dy)
            transect_line = LineString([start_point, transect_end])
            transects.append(transect_line)

    # Create a GeoDataFrame for the transects
    transect_gdf = gpd.GeoDataFrame(geometry=transects, crs=shoreline_gdf.crs)
    # Ensure the CRS is in meters (if not already), for example, using UTM for Tunisia
    transect_gdf = transect_gdf.to_crs(epsg=32632)  # EPSG:32632 is UTM Zone 32N
    # Calculate the length of each LineString and add as a new column in meters
    transect_gdf['length_m'] = transect_gdf.length
    transect_gdf['cumulative_percentile_50'] = cumulative
    transect_gdf.to_file(os.path.join(shapefiles, f"{country}shoreline_one_sided_transects_1_{years_multiplier}.shp"))
    print("Shoreline LineString shapefile saved successfully.")
    return transect_gdf, gdf_sorted



def trasfer_attributes_to_transects(country,shapefiles,  transect_gdf, gdf_sorted,years_multiplier):
    # Load the transects and shoreline points
    # transect_gdf = gpd.read_file()
    shoreline_points_gdf = gdf_sorted # gpd.read_file("shoreline_points.shp")
    # Ensure both GeoDataFrames are in the same CRS
    shoreline_points_gdf = shoreline_points_gdf.to_crs(transect_gdf.crs)
    # Perform a spatial join to transfer point attributes to intersecting transects
    transects_with_attributes = gpd.sjoin(transect_gdf, shoreline_points_gdf, how="left", predicate="intersects")
    transects_with_attributes.to_file(os.path.join(shapefiles, f"{country}shoreline_transects_with_attributes_{years_multiplier}.shp"))
    print(f"transects_with_attributes obs {len(transects_with_attributes)}")
    print("Transect shapefile with point attributes saved successfully.")
    return transects_with_attributes

