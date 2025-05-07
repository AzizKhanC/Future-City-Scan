import xarray as xr
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from osgeo import gdal, osr
import rasterio
from os.path import exists
import math
from pathlib import PureWindowsPath
from pathlib import Path
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio import shutil as rio_shutil

from shutil import copyfile
import shutil

import yaml
# Import OGR and GDAL
from osgeo import gdal, ogr, osr
from osgeo_utils import gdal_calc
from osgeo.gdalconst import GA_Update
import os
import numpy as np
import pandas as pd
# from rasterio import shutil as rio_shutil
from rasterio.vrt import WarpedVRT

# from __future__ import print_function
# Import OGR and GDAL
from osgeo import ogr, gdal
from osgeo_utils import gdal_calc
import rasterio
import numpy as np
import os
import osmnx as ox
import geopandas as gpd


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def preprocess_fathom(output, city, aoi_file, country):
    
    data_folder =Path("data/fathom") / country
    city_name_l=city.replace(' ', '_').lower()
    city_name_l=city_name_l.lower()
    # load configs
    with open(Path("configs.yml"), 'r') as f:
        configs = yaml.safe_load(f)
    # SET UP ##############################################
    # load global inputs, such as data sources that generally remain the same across scans
    # with open(Path("configs.yml"), 'r') as f:
    #     global_inputs = yaml.safe_load(f)

    # Read AOI shapefile --------
    print('read AOI shapefile')
    # transform the input shp to correct prj (epsg 4326)
    features = aoi_file.geometry
    aoi_bounds = aoi_file.bounds

    # Define output folder ---------
    output_path=Path(output) 
    create_dir(output_path)

    flood_folder = Path(output_path) / 'flood'
    create_dir(flood_folder)

    # output_folder = Path(output)/ 'fathom_clean'
    output_folder = Path(output_path)/ 'fathom_clean'
    create_dir(output_folder)

    def tile_finder(direction, tile_size = 1):
        coord_list = []
        if direction == 'lat':
            hemi_options = ['N', 'S']
            coord_min = aoi_bounds.miny
            coord_max = aoi_bounds.maxy
            zfill_digits = 2
        elif direction == 'lon':
            hemi_options = ['E', 'W']
            coord_min = aoi_bounds.minx
            coord_max = aoi_bounds.maxx
            zfill_digits = 3
        else:
            print('tile_finder function error')
            print('Invalid direction. How did this happen?')
        for i in range(len(aoi_bounds)):
            if math.floor(coord_min[i]) >= 0:
                hemi = hemi_options[0]
                for y in range(math.floor(coord_min[i] / tile_size) * tile_size, 
                                math.ceil(coord_max[i] / tile_size) * tile_size, 
                                tile_size):
                    coord_list.append(f'{hemi}{str(y).zfill(zfill_digits)}')
            elif math.ceil(coord_max[i]) >= 0:
                for y in range(0, 
                                math.ceil(coord_max[i] / tile_size) * tile_size, 
                                tile_size):
                    coord_list.append(f'{hemi_options[0]}{str(y).zfill(zfill_digits)}')
                for y in range(math.floor(coord_min[i] / tile_size) * tile_size, 
                                0, 
                                tile_size):
                    coord_list.append(f'{hemi_options[1]}{str(-y).zfill(zfill_digits)}')
            else:
                hemi = hemi_options[1]
                for y in range(math.floor(coord_min[i] / tile_size) * tile_size, 
                                math.ceil(coord_max[i] / tile_size) * tile_size, 
                                tile_size):
                    coord_list.append(f'{hemi}{str(-y).zfill(zfill_digits)}')
        return coord_list


    # Start list of failed processes --------------
    failed = []
    # Prepare flood data (coastal, fluvial, pluvial) ---------------------
    # merged data folder (before clipping)
    # 8 return periods
    # rps = [10, 100, 1000, 20, 200, 5, 50, 500]
    rps = configs['flood']['rps']
    # find relevant tiles
    lat_tiles = tile_finder('lat')
    lon_tiles = tile_finder('lon')
    flood_threshold = configs['flood']['threshold']
    flood_years = configs['flood']['year']
    flood_ssps = configs['flood']['ssp']
    flood_ssp_labels = {1: '1_2.6', 2: '2_4.5', 3: '3_7.0', 5: '5_8.5'}
    flood_prob_cutoff = configs['flood']['prob_cutoff']
    if not len(flood_prob_cutoff) == 2:
        err_msg = '2 cutoffs required for flood'
        print(err_msg)
        failed.append(err_msg)
    else:
        # translate the annual probability cutoffs to bins of return periods
        flood_rp_bins = {f'lt{flood_prob_cutoff[0]}': [], 
                        f'{flood_prob_cutoff[0]}-{flood_prob_cutoff[1]}': [], 
                        f'gt{flood_prob_cutoff[1]}': []}
        for rp in rps:
            annual_prob = 1/rp*100
            if annual_prob < flood_prob_cutoff[0]:
                flood_rp_bins[f'lt{flood_prob_cutoff[0]}'].append(rp)
            elif annual_prob >= flood_prob_cutoff[0] and annual_prob <= flood_prob_cutoff[1]:
                flood_rp_bins[f'{flood_prob_cutoff[0]}-{flood_prob_cutoff[1]}'].append(rp)
            elif annual_prob > flood_prob_cutoff[1]:
                flood_rp_bins[f'gt{flood_prob_cutoff[1]}'].append(rp)

    # Prepare flood data (coastal, fluvial, pluvial) ---------------------
    # city_inputs['country_name']
    if configs['flood_coastal'] or configs['flood_fluvial'] or configs['flood_pluvial']:
        print('prepare flood')
        # 8 return periods
        # rps = [10, 100, 1000, 20, 200, 5, 50, 500]
        rps = configs['flood']['rps']
        # find relevant tiles
        lat_tiles = tile_finder('lat')
        lon_tiles = tile_finder('lon')

        flood_threshold = configs['flood']['threshold']
        flood_years = configs['flood']['year']
        flood_ssps = configs['flood']['ssp']
        flood_ssp_labels = {1: '1_2.6', 2: '2_4.5', 3: '3_7.0', 5: '5_8.5'}
        flood_prob_cutoff = configs['flood']['prob_cutoff']
        if not len(flood_prob_cutoff) == 2:
            err_msg = '2 cutoffs required for flood'
            print(err_msg)
            failed.append(err_msg)
        else:
            # translate the annual probability cutoffs to bins of return periods
            flood_rp_bins = {f'lt{flood_prob_cutoff[0]}': [], 
                            f'{flood_prob_cutoff[0]}-{flood_prob_cutoff[1]}': [], 
                            f'gt{flood_prob_cutoff[1]}': []}
            for rp in rps:
                annual_prob = 1/rp*100
                if annual_prob < flood_prob_cutoff[0]:
                    flood_rp_bins[f'lt{flood_prob_cutoff[0]}'].append(rp)
                elif annual_prob >= flood_prob_cutoff[0] and annual_prob <= flood_prob_cutoff[1]:
                    flood_rp_bins[f'{flood_prob_cutoff[0]}-{flood_prob_cutoff[1]}'].append(rp)
                elif annual_prob > flood_prob_cutoff[1]:
                    flood_rp_bins[f'gt{flood_prob_cutoff[1]}'].append(rp)

            # function to check output flood raster validity
            def flood_raster_check(raster):
                with rasterio.open(raster) as src:
                    return (np.nanmax(src.read(1)) > 1)

            def mosaic_flood_tiles_and_threshold(flood_type):
                print(f'prepare {flood_type} flood')

                # raw data folder
                flood_type_folder_dict = {
                                        # 'coastal': 'COASTAL_UNDEFENDED',
                                        'coastal': 'COASTAL_DEFENDED',
                                        'fluvial': 'FLUVIAL_UNDEFENDED',
                                        'pluvial': 'PLUVIAL_DEFENDED'}
                raw_flood_folder = Path(configs['flood_source']) / f"{country}" / flood_type_folder_dict[flood_type]
                
                # prepare flood raster files (everything before clipping)
                for year in flood_years:
                    if year <= 2020:
                        for rp in rps:
                            # identify tiles and merge as needed
                            raster_to_mosaic = []
                            mosaic_file = f'{city_name_l}_{flood_type}_{year}_1in{rp}.tif'

                            if not exists(flood_folder / mosaic_file):
                                for lat in lat_tiles:
                                    for lon in lon_tiles:
                                        raster_file_name = f"{year}/1in{rp}/1in{rp}-{flood_type_folder_dict[flood_type].replace('_', '-')}-{year}_{lat.lower()}{lon.lower()}.tif"
                                        if exists(raw_flood_folder / raster_file_name):
                                            raster_to_mosaic.append(raw_flood_folder / raster_file_name)
                                if len(raster_to_mosaic) == 0:
                                    print(f'no raster for {flood_type} {year} 1-in-{rp}')
                                elif len(raster_to_mosaic) == 1:
                                    copyfile(raster_to_mosaic[0], flood_folder / mosaic_file)
                                else:
                                    try:
                                        raster_to_mosaic1 = []
                                        for p in raster_to_mosaic:
                                            raster = rasterio.open(p)
                                            raster_to_mosaic1.append(raster)
                                        
                                        mosaic, output = merge(raster_to_mosaic1)
                                        output_meta = raster.meta.copy()
                                        output_meta.update(
                                            {"driver": "GTiff",
                                            "height": mosaic.shape[1],
                                            "width": mosaic.shape[2],
                                            "transform": output,
                                            }
                                        )

                                        with rasterio.open(flood_folder / mosaic_file, 'w', **output_meta) as m:
                                            m.write(mosaic)
                                    except MemoryError:
                                        err_msg = f'MemoryError when merging flood_{flood_type} {year} 1-in-{rp} raster files.'
                                        print(err_msg) 
                                        print('Try GIS instead for merging.')
                                        failed.append(err_msg)
                            
                            # apply threshold
                            if exists(flood_folder / mosaic_file):
                                def flood_con():
                                    with rasterio.open(flood_folder / mosaic_file) as src:
                                        out_image = src.read(1)
                                        out_image[out_image == src.meta['nodata']] = 0
                                        out_image[out_image < flood_threshold] = 0
                                        out_image[out_image >= flood_threshold] = 1
                                        out_meta = src.meta.copy()
                                        out_meta.update({'nodata': 0})

                                    with rasterio.open(flood_folder / f'{mosaic_file[:-4]}_con.tif', "w", **out_meta) as dest:
                                        dest.write(out_image, 1)
                                
                                flood_con()
                                while flood_raster_check(flood_folder / f'{mosaic_file[:-4]}_con.tif'):
                                    flood_con()

                    elif year > 2020:
                        for ssp in flood_ssps:
                            for rp in rps:
                                # identify tiles and merge as needed
                                raster_to_mosaic = []
                                mosaic_file = f'{city_name_l}_{flood_type}_{year}_ssp{ssp}_1in{rp}.tif'

                                if not exists(flood_folder / mosaic_file):
                                    for lat in lat_tiles:
                                        for lon in lon_tiles:
                                            raster_file_name = f"{year}/SSP{flood_ssp_labels[ssp]}/1in{rp}/1in{rp}-{flood_type_folder_dict[flood_type].replace('_', '-')}-{year}-SSP{flood_ssp_labels[ssp]}_{lat.lower()}{lon.lower()}.tif"
                                            if exists(raw_flood_folder / raster_file_name):
                                                raster_to_mosaic.append(raw_flood_folder / raster_file_name)
                                    if len(raster_to_mosaic) == 0:
                                        print(f'no raster for {flood_type} {year} ssp{ssp} 1-in-{rp}')
                                    elif len(raster_to_mosaic) == 1:
                                        copyfile(raster_to_mosaic[0], flood_folder / mosaic_file)
                                    else:
                                        try:
                                            raster_to_mosaic1 = []
                                            for p in raster_to_mosaic:
                                                raster = rasterio.open(p)
                                                raster_to_mosaic1.append(raster)
                                            
                                            mosaic, output = merge(raster_to_mosaic1)
                                            output_meta = raster.meta.copy()
                                            output_meta.update(
                                                {"driver": "GTiff",
                                                "height": mosaic.shape[1],
                                                "width": mosaic.shape[2],
                                                "transform": output,
                                                }
                                            )

                                            with rasterio.open(flood_folder / mosaic_file, 'w', **output_meta) as m:
                                                m.write(mosaic)
                                        except MemoryError:
                                            err_msg = f'MemoryError when merging flood_{flood_type} {year} ssp{ssp} 1-in-{rp} raster files.'
                                            print(err_msg) 
                                            print('Try GIS instead for merging.')
                                            failed.append(err_msg)
                                
                                # apply threshold
                                if exists(flood_folder / mosaic_file):
                                    def flood_con():
                                        with rasterio.open(flood_folder / mosaic_file) as src:
                                            out_image = src.read(1)
                                            out_image[out_image == src.meta['nodata']] = 0
                                            out_image[out_image < flood_threshold] = 0
                                            out_image[out_image >= flood_threshold] = 1
                                            if np.nanmax(out_image) > 1:
                                                print(mosaic_file)
                                                print('max value: ', np.nanmax(out_image))
                                                exit()
                                            out_meta = src.meta.copy()
                                            out_meta.update({'nodata': 0})

                                        with rasterio.open(flood_folder / f'{mosaic_file[:-4]}_con.tif', "w", **out_meta) as dest:
                                            dest.write(out_image, 1)                               
                                    flood_con()
                                    while flood_raster_check(flood_folder / f'{mosaic_file[:-4]}_con.tif'):
                                        flood_con()
            for ft in ['coastal', 'fluvial', 'pluvial']:
                if configs[f'flood_{ft}']:
                    mosaic_flood_tiles_and_threshold(ft)
                    
    print("Fathom mosaiced")
    if configs['flood_coastal'] or configs['flood_fluvial'] or configs['flood_pluvial']:
        # calculate UTM zone from avg longitude to define CRS to project to
        features = aoi_file.geometry
        avg_lng = features.unary_union.centroid.x
        utm_zone = math.floor((avg_lng + 180) / 6) + 1
        utm_crs = f"+proj=utm +zone={utm_zone} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
        # buffer AOI
        buffer_aoi = aoi_file.buffer(np.nanmax([aoi_bounds.maxx - aoi_bounds.minx, aoi_bounds.maxy - aoi_bounds.miny])).geometry
        def reproject_flood_and_bin(flood_type):
            # print(f'process {flood_type} flood')
            for year in flood_years:
                if year <= 2020:
                    for bin in flood_rp_bins:
                        # print(f"bin = {bin}")
                        raster_to_merge = [f'{city_name_l}_{flood_type}_{year}_1in{rp}_con.tif' for rp in flood_rp_bins[bin] if exists(flood_folder / (f'{city_name_l}_{flood_type}_{year}_1in{rp}_con.tif'))]
                        raster_arrays = []
                        # print(f"raster_to_merge--->> {raster_to_merge}")
                        for r in raster_to_merge:
                            # print(f"for each raster ::>>> {r}")
                            with rasterio.open(flood_folder / r) as src:
                                # shapely presumes all operations on two or more features exist in the same Cartesian plane.
                                out_image, out_transform = rasterio.mask.mask(
                                    src, buffer_aoi, all_touched = True, crop = True)
                                out_meta = src.meta.copy()

                            out_meta.update({"driver": "GTiff",
                                            "height": out_image.shape[1],
                                            "width": out_image.shape[2],
                                            "transform": out_transform})
                            
                            raster_arrays.append(out_image)
                            # print(f"r.append(out_image)")
                            # raster_arrays.append(src.read(1))
                            # out_meta = src.meta.copy()
                        
                        if raster_arrays:
                            # print(f" raster_arrays  {raster_arrays}")
                            out_image = np.logical_or.reduce(raster_arrays).astype(np.uint8)
                            out_meta.update(dtype = rasterio.uint8)
                            # print(f"Post--->> np.logical_or.reduce")
                            with rasterio.open(output_folder / f'{city_name_l}_{flood_type}_{year}_{bin}.tif', 'w', **out_meta) as dst:
                                dst.write(out_image)
                            # print(f"Post--->> asterio.open(output_folder ")
                            with rasterio.open(output_folder / f'{city_name_l}_{flood_type}_{year}_{bin}.tif') as src:
                                transform, width, height = calculate_default_transform(
                                    src.crs, utm_crs, src.width, src.height, *src.bounds)
                                kwargs = src.meta.copy()
                                kwargs.update({
                                    'crs': utm_crs,
                                    'transform': transform,
                                    'width': width,
                                    'height': height
                                })
                                # print(f"Pre--->> reproject ")
                                with rasterio.open(output_folder / f'{city_name_l}_{flood_type}_{year}_{bin}_utm.tif', 'w', **kwargs) as dst:
                                    for i in range(1, src.count + 1):
                                        reproject(
                                            source=rasterio.band(src, i),
                                            destination=rasterio.band(dst, i),
                                            src_transform=src.transform,
                                            src_crs=src.crs,
                                            dst_transform=transform,
                                            dst_crs=utm_crs,
                                            resampling=Resampling.nearest)
                elif year > 2020:
                    for ssp in flood_ssps:
                        for bin in flood_rp_bins:
                            raster_to_merge = [f'{city_name_l}_{flood_type}_{year}_ssp{ssp}_1in{rp}_con.tif' for rp in flood_rp_bins[bin] if exists(flood_folder / (f'{city_name_l}_{flood_type}_{year}_ssp{ssp}_1in{rp}_con.tif'))]
                            raster_arrays = []
                            # print(f"Future raster_to_merge--->> {raster_to_merge}")
                            for r in raster_to_merge:
                                with rasterio.open(flood_folder / r) as src:
                                    # shapely presumes all operations on two or more features exist in the same Cartesian plane.
                                    out_image, out_transform = rasterio.mask.mask(
                                        src, buffer_aoi, all_touched = True, crop = True)
                                    out_meta = src.meta.copy()
                                out_meta.update({"driver": "GTiff",
                                                "height": out_image.shape[1],
                                                "width": out_image.shape[2],
                                                "transform": out_transform})
                                
                                raster_arrays.append(out_image)
                                # raster_arrays.append(src.read(1))
                                # out_meta = src.meta.copy()
                            if raster_arrays:
                                out_image = np.logical_or.reduce(raster_arrays).astype(np.uint8)
                                out_meta.update(dtype = rasterio.uint8)
                                with rasterio.open(output_folder / f'{city_name_l}_{flood_type}_{year}_ssp{ssp}_{bin}.tif', 'w', **out_meta) as dst:
                                    dst.write(out_image)
        
                                with rasterio.open(output_folder / f'{city_name_l}_{flood_type}_{year}_ssp{ssp}_{bin}.tif') as src:
                                    transform, width, height = calculate_default_transform(
                                        src.crs, utm_crs, src.width, src.height, *src.bounds)
                                    kwargs = src.meta.copy()
                                    kwargs.update({
                                        'crs': utm_crs,
                                        'transform': transform,
                                        'width': width,
                                        'height': height
                                    })

                                    with rasterio.open(output_folder / f'{city_name_l}_{flood_type}_{year}_ssp{ssp}_{bin}_utm.tif', 'w', **kwargs) as dst:
                                        for i in range(1, src.count + 1):
                                            reproject(
                                                source=rasterio.band(src, i),
                                                destination=rasterio.band(dst, i),
                                                src_transform=src.transform,
                                                src_crs=src.crs,
                                                dst_transform=transform,
                                                dst_crs=utm_crs,
                                                resampling=Resampling.nearest)
            
        for ft in ['coastal', 'fluvial', 'pluvial']:
            if configs[f'flood_{ft}']:
                reproject_flood_and_bin(ft)
        print(f"Fathom processing finished success fully for {city}")
        
        
# A. Population Exposure

def gdal_resample(input_raster, target_pop_raster, ouput_path):
        
        """
        Warp a raster to an inputted resolution.
        
        Args:
            xres (int): output resolution in x-direction
            yres (int): output resolution in y-direction
            ouput_path (str): filepath to where the output file should be stored

        Returns:  Ouputs a raster file inputted resolution.

        """
        ds = gdal.Open(target_pop_raster)
        gt = ds.GetGeoTransform()
        xsize = gt[1]
        ysize = -gt[-1]
        Projection = ds.GetProjectionRef()
        print(f'Projection--{Projection}')
        ds = None
        inDs = gdal.Open(input_raster)
        # inDs=inDs.GetRasterBand(1).SetNoDataValue(nodata)
        props = inDs.GetGeoTransform()
        # print('current pixel xsize:', props[1], 'current pixel ysize:', -props[-1])
        options = gdal.WarpOptions(options=['tr'], xRes=xsize, yRes=ysize, targetAlignedPixels = True, resampleAlg = gdal.GRA_Sum, dstSRS=Projection)          
        # print(f"props-- {props}")
        newfile = gdal.Warp(ouput_path, inDs , options=options, overwrite=True)
        # newfile = gdal.Warp(ouput_path, inDs , options=options)
        newprops = newfile.GetGeoTransform()
        print(f"Original sum: {inDs.GetRasterBand(1).Checksum()}, Warped sum::{newfile.GetRasterBand(1).Checksum()}")
        return ouput_path

def replace_nodata_and_negatives_with_0(file,outfile):
    
    """This function replaces nodata and negatives with zeros

    Parameters
    -------------
    file: str 
        A filename 
    outfile: str 
        outfile filename        

    Returns
    ----------
    The clean filename
    
    """
    ds = gdal.Open(file)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    [rows, cols] = arr.shape
    arr_min = arr.min()
    arr_max = arr.max()
    arr_mean = int(arr.mean())
    print(f"Pre-proccessed: arr_min:{arr_min}  arr_max:{arr_max}  arr_mean:{arr_mean} ")
    # arr_out = np.where((arr < arr_mean), 10000, arr)
    arr_out = np.where((arr < 0), 0, arr)
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(outfile, cols, rows, 1, gdal.GDT_UInt16)
    outdata.SetGeoTransform(ds.GetGeoTransform())##sets same geotransform as input
    outdata.SetProjection(ds.GetProjection())##sets same projection as input
    outdata.GetRasterBand(1).WriteArray(arr_out)
    outdata.GetRasterBand(1).SetNoDataValue(0)##if you want these values transparent
    outdata.FlushCache() ##saves to disk!!
    outdata = None
    band=None
    ds=None
    print(f"Post-proccessed: arr_min:{arr_out.min()}  arr_max:{arr_out.max()}  arr_mean:{arr_out.mean()} ")
    return outfile



def binarize_flood_mosaic(file,outfile, flood_threshold):
    
    """
    This function burns values 1 for all non-zero values

    Parameters
    -------------
    file: str 
        A filename 
    outfile: str 
        outfile filename        

    Returns
    ----------
    The clean filename
    
    """
    ds = gdal.Open(file)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    [rows, cols] = arr.shape
    arr_min = arr.min()
    arr_max = arr.max()
    arr_mean = int(arr.mean())
    print(f"Pre-proccessed: arr_min:{arr_min}  arr_max:{arr_max}  arr_mean:{arr_mean} ")
    # arr_out = np.where((arr < arr_mean), 10000, arr)
    # arr_out = np.where((arr > 0), 1, arr)
    # arr_out = np.where((arr_out < 0), 0, arr_out)
    arr_out=arr
    arr_out[arr_out < flood_threshold] = 0
    arr_out[arr_out >= flood_threshold] = 1

    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(outfile, cols, rows, 1, gdal.GDT_UInt16)
    outdata.SetGeoTransform(ds.GetGeoTransform())##sets same geotransform as input
    outdata.SetProjection(ds.GetProjection())##sets same projection as input
    outdata.GetRasterBand(1).WriteArray(arr_out)
    outdata.GetRasterBand(1).SetNoDataValue(0)##if you want these values transparent
    outdata.FlushCache() ##saves to disk!!
    outdata = None
    band=None
    ds=None
    print(f"Post-proccessed: arr_min:{arr_out.min()}  arr_max:{arr_out.max()}  arr_mean:{arr_out.mean()} ")
    return outfile

def reproject_image_to_master( master, raster_to_be_projected, res=None ):

    """This function reprojects an image (``raster_to_be_projected``) to
    match the extent, resolution and projection of another
    (``master``) using GDAL. The newly reprojected image
    is a GDAL VRT file for efficiency. A different spatial
    resolution can be chosen by specifyign the optional
    ``res`` parameter. The function returns the new file's
    name.

    Parameters
    -------------
    master: str 
        A filename (with full path if required) with the 
        master image (that that will be taken as a reference)
    raster_to_be_projected: str 
        A filename (with path if needed) with the image
        that will be reprojected
    res: float, optional
        The desired output spatial resolution, if different 
        to the one in ``master``.

    Returns
    ----------
    The reprojected filename
    """
    
    raster_to_be_projected_ds = gdal.Open( raster_to_be_projected )
    if raster_to_be_projected_ds is None:
        raise IOError("GDAL could not open raster_to_be_projected file %s " % raster_to_be_projected)
    raster_to_be_projected_proj = raster_to_be_projected_ds.GetProjection()
    raster_to_be_projected_geotrans = raster_to_be_projected_ds.GetGeoTransform()
    data_type = raster_to_be_projected_ds.GetRasterBand(1).DataType
    n_bands = raster_to_be_projected_ds.RasterCount

    master_ds = gdal.Open( master )
    if master_ds is None:
        raise IOError("GDAL could not open master file %s " % master)
    master_proj = master_ds.GetProjection()
    master_geotrans = master_ds.GetGeoTransform()
    w = master_ds.RasterXSize
    h = master_ds.RasterYSize
    if res is not None:
        master_geotrans[1] = float( res )
        master_geotrans[-1] = - float ( res )

    dst_filename = raster_to_be_projected.replace(".tif", "_crop.tif" )
    dst_ds = gdal.GetDriverByName('GTiff').Create(dst_filename,
                                                w, h, n_bands, data_type)
    dst_ds.SetGeoTransform( master_geotrans )
    dst_ds.SetProjection( master_proj)
    gdal.ReprojectImage( raster_to_be_projected_ds, dst_ds, raster_to_be_projected_proj,
                         master_proj, gdal.GRA_NearestNeighbour)
    dst_ds = None  # Flush to disk
    return dst_filename


def convert_raster_array(rasterfn):
    raster = gdal.Open(rasterfn)
    band = raster.GetRasterBand(1)
    return band.ReadAsArray()

def getNoDataValue(rasterfn):
    raster = gdal.Open(rasterfn)
    band = raster.GetRasterBand(1)
    return band.GetNoDataValue()

def convert_array_to_raster(rasterfn,newRasterfn,array):
    raster = gdal.Open(rasterfn)
    geotransform = raster.GetGeoTransform()
    originX = geotransform[0]
    originY = geotransform[3]
    pixelWidth = geotransform[1]
    pixelHeight = geotransform[5]
    cols = raster.RasterXSize
    rows = raster.RasterYSize
    driver = gdal.GetDriverByName('GTiff')
    # outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Float32) #GDT_Int16
    outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Int16 ) 
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(array)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromWkt(raster.GetProjectionRef())
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()
    outband=None
    raster=None
def raster_sum(input_raster):
    with rasterio.open(input_raster, 'r') as ds:
        arr = ds.read()  # read all raster values
        arr[arr < 0] = 0
        actual_sum= np.nansum(arr)
    return actual_sum

def pop_exposure(pop_raster, input_raster, flood_threshold):

    file=input_raster
    outfile= file.replace(".tif", "_clean_delete_me.tif")
    input_raster=binarize_flood_mosaic(file,outfile, flood_threshold)
    out_raster=   input_raster.replace("_clean_delete_me.tif", "upsampled_clean_delete_me.tif")
    upsampled_raster= gdal_resample(input_raster, pop_raster, out_raster)
    # Now we want to replace 0 and nulls with 1s and count them. we want the total number for %area calculation
    rasterfn =input_raster
    newValue = 1
    newRasterfn = out_raster.replace("_clean_delete_me.tif", "_no_data_updated_delete_me.tif") #fr"C:\Users\Aziz\Dropbox\CRP\CS\Resampling task\raster\{file_name}_no_data_updated.tif"
    # Convert Raster to array
    rasterArray = convert_raster_array(rasterfn)
    # Get no data value of array
    noDataValue = getNoDataValue(rasterfn)
    # Updata no data value in array with new value
    rasterArray[rasterArray == noDataValue] = newValue
    # Write updated array to new raster
    convert_array_to_raster(rasterfn,newRasterfn,rasterArray)
    # Now resample the updated no value raster 
    out_raster= newRasterfn.replace("_no_data_updated_delete_me.tif", "_no_data_updated_upsampled_delete_me.tif") # fr"C:\Users\Aziz\Dropbox\CRP\CS\Resampling task\raster\{file_name}_{res}x{res}_no_data_updated.tif"  
    upsampled_raster_b= gdal_resample(newRasterfn, pop_raster, out_raster)

    out_raster_a= upsampled_raster 
    out_raster_b= upsampled_raster_b   
    out_raster_final=out_raster.replace("_no_data_updated_upsampled_delete_me.tif", "_divided_delete_me.tif") #  fr"C:\Users\Aziz\Dropbox\CRP\CS\Resampling task\raster\{file_name}_{res}x{res}_no_data_updated_divided.tif"  
    out0=gdal_calc.Calc(calc="A/B*100", A=out_raster_a , B=out_raster_b, outfile=out_raster_final, type="Int16", overwrite=True, extent="intersect", NoDataValue=0)

    # Number of people expose
    out_raster_a= pop_raster  
    out_raster_b= out_raster_final   
    base_path = Path(file).parents[1]
    file_stem = Path(file).stem
    base_path = os.path.join(base_path, "Population_exposure")
    create_dir_recursive(base_path)
    out_raster_final=str(Path(os.path.join(base_path, f"{file_stem}.tif")))
    out_raster_final=out_raster_final.replace("max_exposure_", "final_population_exposure_")
    
    out=gdal_calc.Calc(calc="A*B/100", A=out_raster_a , B=out_raster_b, outfile=out_raster_final, type="Int16", overwrite=True, extent="intersect", NoDataValue=0)
    pop_rasterArray = convert_raster_array(pop_raster)
    band = out.GetRasterBand(1)
    arr = band.ReadAsArray()
    arr_min = arr.min()
    arr_max = arr.max()
    arr_mean = int(arr.mean())
    arr_sum = int(arr.sum())
    # initialize data of lists.
    data = {
            'Total  population':int(pop_rasterArray.sum()),
            'Population min' : pop_rasterArray.min(),
            'Population max' : pop_rasterArray.max(),
            'Population mean': int(pop_rasterArray.mean()),
            'Total Exposed population': int(arr.sum()),
            'Exposed population min': arr.min(), 
            'Exposed population max': arr.max(), 
            'Exposed population mean':  int(arr.mean()),
            '%Exposed population':  int((arr.sum()/pop_rasterArray.sum())*100)

            }
    # Create DataFrame
    df = pd.DataFrame([data])
    # Print the output.
    print(df)
    out0=None
    out=None
    band=None
    raster_delete=[outfile, out_raster, upsampled_raster, input_raster, newRasterfn,upsampled_raster_b]
    for fname in raster_delete:
        try:
            if os.path.isfile(fname): 
                os.remove(fname)
        except Exception as e:
            print(f"{fname} couldnt be deleted... {e}")

    return df, out_raster_final


###

def create_dir_recursive(directory):
    os.makedirs(directory, exist_ok = True) 
    return directory


def crop_to_shp(path_name, df):
    raster =rasterio.open(path_name)
    # Mask to shapefile
    out_array, out_trans = rasterio.mask.mask(dataset=raster, shapes=df.geometry, crop=True)
    out_meta = raster.meta.copy()
    out_meta.update({"height": out_array.shape[1], 
                      "width": out_array.shape[2],
                      "transform": out_trans})
    return out_array, out_meta

def export_array_to_raster(array, meta, path, fileName):
    Cropped_outputfile = path / fileName
    with rasterio.open(Cropped_outputfile, 'w', **meta, compress = 'LZW', tiled=True) as dest:
        dest.write(array)
    return Cropped_outputfile    


def multiply_array_list(arrayList, array):
      flood_pop = []
      for i in range(len(arrayList)):
          flood_pop.append(np.multiply(arrayList[i], array))
      flood_pop = (np.vstack((flood_pop)))
      #tPrint(f"{countryCode}_{adm1Index} - Flood multiplied by pop")
      return flood_pop

def convert_to_bool_int_array(array, numberCategories):
      listNumberCat = range(numberCategories)
      list_IntBool_floodCat = []
      # Create a boolean int array for each category
      for i in listNumberCat:
          intArray = (array == i).astype(dtype = np.int8, copy = False)
          list_IntBool_floodCat.append(intArray)
      #tPrint(f"{countryCode}_{adm1Index} - Flood converted to bool int arrays")
      return list_IntBool_floodCat

def vrtWarp(inpath, outpath, vrt_settings, city, filePrefix):
    with rasterio.open(inpath) as src:
      with WarpedVRT(src, **vrt_settings) as vrt:
          data = vrt.read()
          # Process the dataset in chunks. 
          for _, window in vrt.block_windows(): data = vrt.read(window=window)
          # Export the aligned data 
          fileName = f"{city}_{filePrefix}_warped.tif"
          Aligned_outputfile = outpath / fileName        
          rio_shutil.copy(vrt, Aligned_outputfile, driver='GTiff')
    return Aligned_outputfile

def array3d_sum(array):
      Flood_pop_sums = {
        '0-NoRiskPop': np.sum(array[0]),
        '1-LowRiskPop': np.sum(array[1]),
        '2-ModerateRiskPop': np.sum(array[2]),
        '3-HighRiskPop': np.sum(array[3]),
        '4-VeryHighRiskPop': np.sum(array[4]),
        '5-WaterBodyPop': np.sum(array[5])
      }
      #tPrint(f"{countryCode}_{adm1Index} - Checksum: {np.sum(array)}")
      return Flood_pop_sums

def assess_flood_exposure(fluvial_raster,pluvial_raster,coastal_raster,exposure_var_raster,return_period,ssp,Year , raster_keyword, fathom_alignment_folder, subdir,city_vector, flood_bins, numberCategories, city, country):
    delete_me= []
    ###################### Fluvial adm1 ######################
    # crop
    floodArray, out_meta = crop_to_shp(fluvial_raster, city_vector)
    # export
    flood_type="Fluvial"
    output_folder=Path(create_dir_recursive(f"{fathom_alignment_folder}/{flood_type}/{Year}/{ssp}/{flood_type}_{ssp}_{Year}"))
    delete_me.append(output_folder)
    raster_name = f"cropped_{flood_type}_{ssp}_{Year}_1in{return_period}{raster_keyword}.tif"
    FluvialCropped_outputfile = export_array_to_raster(floodArray, out_meta, output_folder, raster_name)
    # ###################### Pluvial city_vector ######################
    # crop
    floodArray, out_meta = crop_to_shp(pluvial_raster, city_vector)
    # export
    flood_type="Pluvial"
    output_folder=Path(create_dir_recursive(f"{fathom_alignment_folder}/{flood_type}/{Year}/{ssp}/{flood_type}_{ssp}_{Year}"))
    delete_me.append(output_folder)
    raster_name = f"cropped_{flood_type}_{ssp}_{Year}_1in{return_period}{raster_keyword}.tif"
    PluvialCropped_outputfile = export_array_to_raster(floodArray, out_meta, output_folder, raster_name)

    # ###################### Coastal Flood This is duplication. delete it######################
    # crop
    floodArray, out_meta = crop_to_shp(coastal_raster, city_vector)
    # export
    flood_type="Coastal"
    output_folder=Path(create_dir_recursive(f"{fathom_alignment_folder}/{flood_type}/{Year}/{ssp}/{flood_type}_{ssp}_{Year}"))
    delete_me.append(output_folder)
    raster_name = f"cropped_{flood_type}_{ssp}_{Year}_1in{return_period}{raster_keyword}.tif"
    CoastalCropped_outputfile = export_array_to_raster(floodArray, out_meta, output_folder, raster_name)

    ###################### WarpedVRT ######################
    # Vritually warp the pop and coastal flood files to the fluvial and pluvial
    vrt_options ={
        # 'resampling': Resampling.nearest,
        'resampling': Resampling.sum,
        'crs': out_meta['crs'],
        'transform' : out_meta['transform'],
        'height':floodArray.shape[1],
        'width': floodArray.shape[2]
    }

    ####################################
    # Get prewarping sum
    actual_pop=raster_sum(exposure_var_raster)
    # print(f"Actual prewarp pop: {actual_pop}")

    ###################### Population ######################
    # Get population file
    flood_type_1="Population"
    flood_type="exposure"
    output_folder=Path(create_dir_recursive(f"{fathom_alignment_folder}/{flood_type}/{Year}/{ssp}/{flood_type_1}_{ssp}_{Year}"))
    raster_name = f"cropped_{flood_type_1}_{ssp}_{Year}_1in{return_period}{raster_keyword}.tif"
    # Align using Warped VRT
    popAligned_outputfile = vrtWarp(exposure_var_raster, output_folder, vrt_options, city, raster_name)
    # crop
    popArray, out_meta = crop_to_shp(popAligned_outputfile, city_vector)
    # change dtype to reduce memory
    popArray = popArray.astype('float32', copy=False)
    out_meta.update({'dtype': 'float32'})
    # export
    raster_name = f"cropped_final_{flood_type}_{ssp}_{Year}_1in{return_period}{raster_keyword}.tif"
    export_array_to_raster(popArray, out_meta, output_folder, raster_name)
    # Remove negatives
    popArray[popArray < 0] = 0
    total_pop=np.sum(popArray)
    # Print population sum for country to compare against final results
    # tPrint(f"{city}- {subdir} aligned & cropped. Total Pop: {total_pop}")
    scaling_factor=actual_pop/total_pop
    # adjust  warped population 
    popArray = popArray*scaling_factor
    # export the rescaled population
    raster_name = f"final_rescaled_warped_and_cropped_{flood_type}_{ssp}_{Year}_1in{return_period}{raster_keyword}.tif"
    export_array_to_raster(popArray, out_meta, output_folder, raster_name)

    ###################### Coastal city_vector ######################
    flood_type="Coastal"
    output_folder=Path(create_dir_recursive(f"{fathom_alignment_folder}/{flood_type}/{Year}/{ssp}/{flood_type}_{ssp}_{Year}"))
    raster_name = f"cropped_warped_{flood_type}_{ssp}_{Year}_1in{return_period}{raster_keyword}.tif"
    # export
    cFloodAligned_outputfile = vrtWarp(coastal_raster, output_folder,  vrt_options, city, raster_name)
    # At this point the coastal flood file is aligned and has been cropped to the pop bounding box
    # It will be cropped to the city_vector boundaries after merging with the fluvail and pluvial files
    ###################### Flood mosaic city_vector ######################
    # Create flood mosaic
    combinedFloodList = [FluvialCropped_outputfile] + [PluvialCropped_outputfile] + [cFloodAligned_outputfile]            
    allFiles = []
    for fp in combinedFloodList:
        src = rasterio.open(fp)
        allFiles.append(src)

    # Merge flood files. Using max pixel by pixel method
    floodArray, out_trans = merge(allFiles, method='max')
    out_meta = allFiles[0].meta.copy()
    out_meta.update({"height": floodArray.shape[1], 
                        "width": floodArray.shape[2],
                        "transform": out_trans})

    # export
    flood_type="exposure"
    output_folder=Path(create_dir_recursive(f"{fathom_alignment_folder}/{flood_type}/{Year}/{ssp}/{flood_type}_{ssp}_{Year}"))
    raster_name = f"mosaiced_max_{flood_type}_{ssp}_{Year}_1in{return_period}{raster_keyword}.tif"
    FloodMerged_outputfile = export_array_to_raster(floodArray, out_meta, output_folder, raster_name)

    # ###################### Crop merged file flood ######################
    # crop
    floodArray, out_meta = crop_to_shp(FloodMerged_outputfile, city_vector)
    # export
    output_folder=Path(create_dir_recursive(f"{fathom_alignment_folder}/{flood_type}/{Year}/{ssp}/{flood_type}_{ssp}_{Year}"))
    raster_name = f"max_{flood_type}_{ssp}_{Year}_1in{return_period}{raster_keyword}.tif"
    FloodCropped_outputfile = export_array_to_raster(floodArray, out_meta, output_folder, raster_name)
    exp_df=0
    

    return exp_df , FloodCropped_outputfile

###

def conduct_pop_exposure(output,tables,rasters, vector_file, revised_bins_list, flood_bins, numberCategories, city, country):
    flood_folder = Path(output) / "flood"
    city_name_l=city.replace(' ', '_').lower()
    city_name_l=city_name_l.lower()
    # SET UP ##############################################
    # load global inputs, such as data sources that generally remain the same across scans
    with open(Path("configs.yml"), 'r') as f:
        configs = yaml.safe_load(f)
    # 8 return periods
    # rps = [10, 100, 1000, 20, 200, 5, 50, 500]
    rps = configs['flood']['rps']
    flood_years = configs['flood']['year']
    flood_ssps = configs['flood']['ssp']
    flood_ssp_labels = {1: '1_2.6', 2: '2_4.5', 3: '3_7.0', 5: '5_8.5'}
    fathom_alignment_folder=Path(output) 
    create_dir_recursive(fathom_alignment_folder)
    raster_keyword=""
    raster_keyword_null=""
    df= []
    # revised_bins_list = [15,30,50,100] 
    df_revised= []
    for year in flood_years:
            if year <= 2020:
                for rp in rps:
                    try:
                        ssp_for_pop="SSP1"
                        # df = []
                        exposure_rasters_list=[]
                        return_period=rp #"merged_threshold"            
                        fluvial_raster=Path(flood_folder / f'{city_name_l}_fluvial_{year}_1in{rp}{raster_keyword}.tif') #Path(f"{rasters}\projected_clipped_fathom_1in{return_period}-Fluvial-DEFENDED-{Year}-{ssp}.tif")
                        pluvial_raster=Path(flood_folder / f'{city_name_l}_pluvial_{year}_1in{rp}{raster_keyword}.tif')
                        coastal_raster=Path(flood_folder / f'{city_name_l}_coastal_{year}_1in{rp}{raster_keyword}.tif')                        
                        
                        ssp_exposure=ssp_for_pop 
                        subdir_for_fathom="popdynamics"
                        exposure_var_raster=Path(f"{rasters}\processed_{subdir_for_fathom}_{ssp_exposure}_{year}.tif")                    
                        # try:
                        # print(f"Run exposure ")
                        city_vector=vector_file                                              
                        exposure_df , exported_raster =assess_flood_exposure(fluvial_raster,pluvial_raster,coastal_raster,exposure_var_raster,return_period,ssp_for_pop,year , 
                                                                             raster_keyword, fathom_alignment_folder, subdir_for_fathom ,city_vector, flood_bins, 
                                                                             numberCategories, city, country)
                        exposure_rasters_list.append(exported_raster)                    
                        for threshold in revised_bins_list:
                            pop_raster=str(exposure_var_raster) #f"{rasters}\processed_popdynamics_SSP1_2030.tif"
                            input_raster=str(exported_raster)
                            df2, out_raster_final=pop_exposure(pop_raster,input_raster, threshold)
                            df2["Population raster name"]=exposure_var_raster.stem
                            df2["Flood raster name"]=exported_raster.stem
                            df2[f"Flood threshold"]=threshold
                            df2["Year"] = year
                            df2["SSP"] = ssp_for_pop
                            df2["Return period"] = rp
                            df_revised.append(df2)
                    except Exception as e:
                        print("Failed---->> {} ".format(e))
                        
            elif year > 2020:
                for ssp in flood_ssps:
                    for rp in rps:
                        try:
                            ssp_for_pop= f"SSP{ssp}".upper() #"SSP1"
                            # df = []
                            exposure_rasters_list=[]
                            return_period=rp #"merged_threshold"            
                            fluvial_raster=Path(flood_folder / f'{city_name_l}_fluvial_{year}_ssp{ssp}_1in{rp}{raster_keyword}.tif') #Path(f"{rasters}\projected_clipped_fathom_1in{return_period}-Fluvial-DEFENDED-{Year}-{ssp}.tif")
                            pluvial_raster=Path(flood_folder / f'{city_name_l}_pluvial_{year}_ssp{ssp}_1in{rp}{raster_keyword}.tif')
                            coastal_raster=Path(flood_folder / f'{city_name_l}_coastal_{year}_ssp{ssp}_1in{rp}{raster_keyword}.tif')                        
                            
                            ssp_exposure=ssp_for_pop 
                            subdir_for_fathom="popdynamics"
                            exposure_var_raster=Path(f"{rasters}\processed_{subdir_for_fathom}_{ssp_exposure}_{year}.tif")                    

                            exposure_df , exported_raster =assess_flood_exposure(fluvial_raster,pluvial_raster,coastal_raster,exposure_var_raster,
                                                                                return_period,ssp_for_pop,year , raster_keyword, fathom_alignment_folder, subdir_for_fathom ,
                                                                                city_vector, flood_bins, numberCategories, city, country)
                            
                            exposure_rasters_list.append(exported_raster)
                            for threshold in revised_bins_list:
                                pop_raster=str(exposure_var_raster) #f"{rasters}\projected_clipped_popdynamics_SSP1_2030.tif"
                                input_raster=str(exported_raster)
                                df2, out_raster_final=pop_exposure(pop_raster,input_raster, threshold)
                                df2["Population raster name"]=exposure_var_raster.stem
                                df2["Flood raster name"]=exported_raster.stem
                                df2["Flood threshold"]=threshold
                                df2["Year"] = year
                                df2["SSP"] = ssp_for_pop
                                df2["Return period"] = rp
                                df_revised.append(df2)

                        except Exception as e:
                            print("Failed---->> {} ".format(e))

    df2 =  pd.concat(df_revised, ignore_index=True)
    df2.to_csv(f"{tables}/{city}_population_exposure_to_flood_panel_revised.csv")
    shutil.rmtree(flood_folder)
    return df2

def fathom_barchart_pop_exposure(df,subtitle, y_var, x_var, cmap, maps, year, i):
    df['Year'] = df.Year.map(str) 
    df['scenario_year']=df['SSP'] + ""
    # Draw a nested barplot by flood-type and ssp
    g = sns.catplot(
        data=df, kind="bar",
        x=x_var, y=y_var, hue='scenario_year',
        palette=cmap, alpha=.9, height=5,
        errorbar=None
    )
    # 
    g.fig.suptitle(f"{subtitle}")
    # title
    new_title = 'Scenario'
    g._legend.set_title(new_title)
    y_var=y_var.replace(":", "_")
    g.figure.savefig(f"{maps}/fathom_barchart_{y_var}_barchart_{year}_{i}.png", dpi=300, bbox_inches='tight')

def create_pop_exposure_barchart(df2, revised_bins_list,cmap, maps):
    # 
    # cmap="cool"
    # cmap='viridis'
    # cmap="rainbow"
    # cmap="hsv"
    # cmap='seismic'
    df4=df2[['%Exposed population','Year','SSP', 'Return period', 'Flood threshold']]
    index_cols = [col for col in df4.columns if col not in ["Flood threshold",'%Exposed population']]
    df3 = df4.pivot(index=index_cols, columns="Flood threshold", values="%Exposed population").add_prefix('Flood depth(cm):').reset_index()
    df3 = df3.rename_axis(None, axis=1)

    x_var="Return period"
    df=df3 # pd.read_csv(f"{tables}/{city}_road_network_fathom_exposure_mosaic_max_panel_gdal.csv")
    # for i in range(15, 400, 35):
    for i in revised_bins_list:
        y_var= f"Percentage of exposed po more than {i}cm"
        y_var= f'Flood depth(cm):{i}'
        for year in df.Year.unique():
            df_s = df.loc[df.Year == year]
            subtitle=f"Percentage of population exposed \n to a {i}cm flood in year {year}"
            # fathom_barchart_pop_exposure(df,subtitle, y_var, x_var ,cmap, maps, year, i)
            fathom_barchart_pop_exposure(df_s,subtitle, y_var, x_var ,cmap, maps, year, i)
            
        
        


# B. Infra/Roads exposure
def export_roads(vector_file, shapefiles ,city ):

    """
    Outputs roads shapefile and gpkp
    
    Args:
        vector_file: Path to input read city shapefile ( As vector)
        shapefiles: Output path for shapefile export 
        road raster.
        flood_bins: flood bins for classes
        city: City

    Returns:
         City road network edges  
    """

    index=0
    polygon = vector_file.reset_index().iloc[index]['geometry']
    G = ox.graph.graph_from_polygon(polygon, network_type='drive')
    # add travel time based on maximum speed
    G = ox.add_edge_speeds(G) 
    G = ox.add_edge_travel_times(G) 
    G = ox.projection.project_graph(G, to_crs=3857)
    # get edges as Geo data frame
    nodes, gdf_edges = ox.graph_to_gdfs(G)
    # Export route edge geometries as shapefile. Ignore nodes 
    gdf_edges_reset=gdf_edges.reset_index()
    edges = gdf_edges_reset.set_index(['u', 'v', 'key'])
    geoms = edges['geometry']
    gdf2 = gpd.GeoDataFrame(geometry=(geoms))
    gdf2.to_file(f'{shapefiles}/{city}_road_network_shapefile.shp')
    # save graph as a geopackage
    roads_filepath=f'{shapefiles}/{city}_road_network_graph_gpkg.gpkg'
    ox.save_graph_geopackage(G, filepath=roads_filepath)
    print(f" {city} road network exported")
    return gdf_edges


def assess_road_network_exposure_to_floods(roads_filepath, fluvial_raster, out_raster_path, roads_with_flood_depth):
    
    """
    Returns path to flooded road network 

    Args:
        roads_filepath: Path to input roads shapefile 
        fluvial_raster: Path to input flood raster 
        out_raster_path: Outpath for rasterized roads 
        roads_with_flood_depth:Outpath for flooded roads 

    Returns:
        Outpath for flooded roads 

    """

    # Open  roads
    dataset = ogr.Open(roads_filepath)
    # Make sure the dataset exists or print error
    if not dataset:
        print('Error: could not open roads')


    ### How many layers are contained in this Shapefile?
    layer_count = dataset.GetLayerCount()
    # print('The shapefile has {n} layer(s)\n'.format(n=layer_count))
    ### What is the name of the 1 layer?
    layer = dataset.GetLayerByIndex(0)
    # print('The layer is named: {n}\n'.format(n=layer.GetName()))

    # Benchmark raster for osr and ref res etc
    raster_ds = gdal.Open(str(fluvial_raster), gdal.GA_ReadOnly)

    if not raster_ds:
        print(f'Error: could not open {str(fluvial_raster)}')

    # Fetch number of rows and columns
    ncol = raster_ds.RasterXSize
    nrow = raster_ds.RasterYSize
    # Fetch projection and extent
    proj = raster_ds.GetProjectionRef()
    ext = raster_ds.GetGeoTransform()
    raster_ds = None
    # Create the raster dataset
    memory_driver = gdal.GetDriverByName('GTiff')
    out_raster_ds = memory_driver.Create(out_raster_path, ncol, nrow, 1, gdal.GDT_Byte)
    # Set the ROI image's projection and extent to our input raster's projection and extent
    out_raster_ds.SetProjection(proj)
    out_raster_ds.SetGeoTransform(ext)
    # Fill our output band with the 0 blank, no class label, value
    b = out_raster_ds.GetRasterBand(1)
    b.Fill(0)
    field_for_burning='length_m' 
    # Rasterize the roads shapefile layer into roads raster
    status = gdal.RasterizeLayer(out_raster_ds,  # output to our new dataset
                                [1],  # output to our new dataset's first band
                                layer,  # rasterize this layer
                                None, None,  # don't worry about transformations since we're in same projection
                                [1], # burn value 1 in all linestrings. replace the attribute in optons if needed by classes or vals
                                ['ALL_TOUCHED=TRUE'  # rasterize all pixels touched by polygons
                                # ,  'ATTRIBUTE=length_m'
                                ]  # put raster values according to the 'id' field values
                                )

    # Close dataset
    out_raster_ds = None
    if status != 0:
        print("Nope rasterization failed" )
    else:
        # print("Successfully rasterized the roads. Now we are subetting the flooded roads and assigning the flood depth as vals")
        gdal_calc.Calc(calc="A*logical_and(A>0,B>0)", A=fluvial_raster, B=out_raster_path, outfile=roads_with_flood_depth , overwrite=True)
        return roads_with_flood_depth


def get_flooded_roads_array(rasterPath, band = 1):
    """
    Returns raster band as 2d numpy array

    Args:
        rasterPath: Path to input raster

    Returns:
        numpy array if band exists or None if band does not exist

    """
    if os.path.isfile(rasterPath):
        ds = gdal.Open(rasterPath)
        bands = ds.RasterCount
        if band > 0 and band <= bands:
            array = ds.GetRasterBand(band).ReadAsArray()
            ds = None
            return array
        else:
            return None
    else:
        return None
    

def reclassif_raster_by_bins(roads_with_flood_depth, roads_with_flood_depth_classified, flood_bins):
    """
    Outputs a reclassified raster 

    Args:
        roads_with_flood_depth: Path to input road raster
        roads_with_flood_depth_classified: Path to output classified 
        road raster.
        flood_bins: flood bins for classes

    Returns:
         None 
    """
    with rasterio.open(roads_with_flood_depth) as src:    
        array = src.read()
        profile = src.profile
        bins = np.array(flood_bins) 
        inds = np.digitize(array, bins, right=True) # True to include right bin edge
    with rasterio.open(roads_with_flood_depth_classified, 'w', **profile) as dst:
        dst.write(inds)
        

###########

# # %%capture
# import road_fathom_exposure as roads_exposure
def conduct_infra_exposure_analysis(output, tables ,shapefiles, flood_bins,flood_years, rps ,flood_ssps,   city, country):
    # fathom_alignment_folder=Path(output) / "fathom_alignment"
    fathom_alignment_folder=Path(output) 
    create_dir_recursive(fathom_alignment_folder)
    infra_exposure_dir= Path(fathom_alignment_folder / "infra_exposure")
    # Note: 0 contains flood depths of 0 up to 0.0000000001 -> no flood
    numberCategories = len(flood_bins)
    raster_keyword=""
    output_df = pd.DataFrame()
    for year in flood_years:
            if year <= 2020:
                for rp in rps:
                    # try:
                    ssp_for_pop="SSP1"
                    df = []
                    exposure_rasters_list=[]
                    return_period=rp #"merged_threshold"            
                    # floodName = f"max_{subdir}_{city}_{ssp}_{Year}_1in{return_period}{raster_keyword}.tif"
                    flood_type="exposure"
                    output_folder=f"{fathom_alignment_folder}/{flood_type}/{year}/{ssp_for_pop}/{flood_type}_{ssp_for_pop}_{year}"
                    flood_fn = Path(output_folder) / f"max_{flood_type}_{ssp_for_pop}_{year}_1in{return_period}{raster_keyword}.tif"
                    raster_stem=flood_fn.stem
                    roads_filepath=f'{shapefiles}/{city}_road_network_shapefile.shp'
                    # roads exposed folder
                    infra_output_folder=f"{fathom_alignment_folder}/{flood_type}/{year}/{ssp_for_pop}/Roads_exposed_{flood_type}_{ssp_for_pop}_{year}"
                    create_dir_recursive(infra_output_folder)
                    out_raster_path= str(Path(infra_output_folder) / f"roads_rasterized_{str(raster_stem)}")
                    roads_with_flood_depth=str(Path(infra_output_folder) / f"roads_flood_depth{str(raster_stem)}" )
                    ###
                    base_path = Path(roads_with_flood_depth).parents[1]
                    file_stem = Path(roads_with_flood_depth).stem
                    base_path = os.path.join(base_path, "Roads_exposure")
                    create_dir_recursive(base_path)
                    out_raster_final=str(Path(os.path.join(base_path,  f"{file_stem}.tif")))
                    out_raster_final=out_raster_final.replace("roads_flood_depth", "final")
                    out_raster_final=out_raster_final.replace("finalmax_exposure_", "final_road_exposure_")
                    roads_with_flood_depth=out_raster_final
                    
                    
                    roads_with_flood_depth_classified=str(Path(infra_output_folder) / f"roads_classified{str(raster_stem)}") 
                    assess_road_network_exposure_to_floods(roads_filepath,flood_fn, out_raster_path, roads_with_flood_depth)
                    rasterPath=roads_with_flood_depth
                    get_flooded_roads_array_flooded_roads= get_flooded_roads_array(rasterPath, band = 1)
                    get_flooded_roads_array_of_all_roads= get_flooded_roads_array(out_raster_path, band = 1)
                    all_roads_length=np.count_nonzero(get_flooded_roads_array_of_all_roads)*30
                    flooded_roads_length=np.count_nonzero(get_flooded_roads_array_flooded_roads)*30
                    print(f"all_roads_length: {all_roads_length}-- flooded_roads_length:{flooded_roads_length}. Fraction={flooded_roads_length/all_roads_length}")
                    reclassif_raster_by_bins(roads_with_flood_depth, roads_with_flood_depth_classified, flood_bins)

                    dicts = {}
                    dicts[f"flood_file_name"] = f" flood_file_name {raster_stem}"
                    dicts[f"Year"] = year
                    dicts[f"SSP"] = ssp_for_pop
                    dicts[f"Return period"] = rp
                    # for i in range(15, 400, 35):
                    for i in flood_bins:
                        flooded_more_than_cm= (((i < get_flooded_roads_array_flooded_roads) & (get_flooded_roads_array_flooded_roads < 1000000)).sum())*30
                        dicts[f"Road length flooded more than {i}cm"] = flooded_more_than_cm
                        frac_flooded_more_than_cm=  (flooded_more_than_cm / all_roads_length)*100
                        dicts[f"Percentage of road flooded more than {i}cm"] = frac_flooded_more_than_cm
                    output_df = pd.concat([output_df, pd.DataFrame([dicts])], ignore_index=True)

                    # except Exception as e:
                    #     print("Failed---->> {} ".format(e))

            elif year > 2020:
                for ssp in flood_ssps:
                    for rp in rps:
                        try:
                            ssp_for_pop= f"SSP{ssp}".upper() #"SSP1"
                            df = []
                            exposure_rasters_list=[]
                            return_period=rp #"merged_threshold"            
                            ssp_exposure=ssp_for_pop 
                            flood_fn= Path(fathom_alignment_folder) / f"max_popdynamics_{city}_{ssp_for_pop}_{year}_1in{rp}{raster_keyword}tif"                         
                            raster_stem=flood_fn.stem
                            roads_filepath=f'{shapefiles}/{city}_road_network_shapefile.shp'
                
                            flood_type="exposure"
                            output_folder=f"{fathom_alignment_folder}/{flood_type}/{year}/{ssp_for_pop}/{flood_type}_{ssp_for_pop}_{year}"
                            flood_fn = Path(output_folder) / f"max_{flood_type}_{ssp_for_pop}_{year}_1in{return_period}{raster_keyword}.tif"
                            raster_stem=flood_fn.stem
                            roads_filepath=f'{shapefiles}/{city}_road_network_shapefile.shp'
                            # roads exposed folder
                            infra_output_folder=f"{fathom_alignment_folder}/{flood_type}/{year}/{ssp_for_pop}/Roads_exposed_{flood_type}_{ssp_for_pop}_{year}"
                            create_dir_recursive(infra_output_folder)
                            out_raster_path= str(Path(infra_output_folder) / f"roads_rasterized_{str(raster_stem)}")
                            roads_with_flood_depth=str(Path(infra_output_folder) / f"roads_flood_depth{str(raster_stem)}" )
                            
                            ###
                            base_path = Path(roads_with_flood_depth).parents[1]
                            file_stem = Path(roads_with_flood_depth).stem
                            base_path = os.path.join(base_path, "Roads_exposure")
                            create_dir_recursive(base_path)
                            out_raster_final=str(Path(os.path.join(base_path,  f"{file_stem}.tif")))
                            out_raster_final=out_raster_final.replace("roads_flood_depth", "final")
                            out_raster_final=out_raster_final.replace("finalmax_exposure_", "final_road_exposure_")
                            roads_with_flood_depth=out_raster_final
                            roads_with_flood_depth_classified=str(Path(infra_output_folder) / f"roads_classified{str(raster_stem)}") 

                            assess_road_network_exposure_to_floods(roads_filepath,flood_fn, out_raster_path, roads_with_flood_depth)
                            rasterPath=roads_with_flood_depth
                            get_flooded_roads_array_flooded_roads= get_flooded_roads_array(rasterPath, band = 1)
                            get_flooded_roads_array_of_all_roads= get_flooded_roads_array(out_raster_path, band = 1)
                            all_roads_length=np.count_nonzero(get_flooded_roads_array_of_all_roads)*30
                            flooded_roads_length=np.count_nonzero(get_flooded_roads_array_flooded_roads)*30
                            # print(f"all_roads_length: {all_roads_length}-- flooded_roads_length:{flooded_roads_length}. Fraction={flooded_roads_length/all_roads_length}")
                            reclassif_raster_by_bins(roads_with_flood_depth, roads_with_flood_depth_classified, flood_bins)

                            dicts = {}
                            dicts[f"flood_file_name"] = f" flood_file_name {raster_stem}"
                            dicts[f"Year"] = year
                            dicts[f"SSP"] = ssp_for_pop
                            dicts[f"Return period"] = rp
                            # for i in range(15, 400, 35):
                            for i in flood_bins:
                                flooded_more_than_cm= (((i < get_flooded_roads_array_flooded_roads) & (get_flooded_roads_array_flooded_roads < 1000000)).sum())*30
                                dicts[f"Road length flooded more than {i}cm"] = flooded_more_than_cm
                                frac_flooded_more_than_cm=  (flooded_more_than_cm / all_roads_length)*100
                                dicts[f"Percentage of road flooded more than {i}cm"] = frac_flooded_more_than_cm
                            output_df = pd.concat([output_df, pd.DataFrame([dicts])], ignore_index=True)

                        except Exception as e:
                            print("Failed---->> {} ".format(e))

    output_df.to_csv(f"{tables}/{city}_road_network_fathom_exposure_mosaic_max_panel_gdal.csv")


def fathom_barchart_infra_exposure(df,subtitle, y_var, x_var, cmap, maps, year, i):
    # df['scenario_year']=df['SSP']+ " ("+str(df['Year'])+")"
    df['Year'] = df.Year.map(str) 
    df['scenario_year']=df['SSP'] + ""
    # Draw a nested barplot by flood-type and ssp
    g = sns.catplot(
        data=df, kind="bar",
        x=x_var, y=y_var, hue='scenario_year',
        palette=cmap, alpha=.9, height=6, 
        errorbar=None
    )
    # 
    g.fig.suptitle(f"{subtitle}")
    # title
    new_title = 'Scenario'
    g._legend.set_title(new_title)
    # sns.move_legend(g, "upper right", bbox_to_anchor=(.8, 1.02), frameon=False)
    # g.set_axis_labels(f"{x_var}", "Roads flooded ")
    g.figure.savefig(f"{maps}/fathom_barchart_{y_var}_barchart_{year}_{i}.png", dpi=300, bbox_inches='tight')

def export_infra_exposure_barcharts(tables, revised_bins_list, cmap, maps,city, country):
    x_var="Return period"
    # y_var="Percentage of road flooded more than 15cm"
    df= pd.read_csv(f"{tables}/{city}_road_network_fathom_exposure_mosaic_max_panel_gdal.csv")
    # for i in range(15, 400, 35):
    for i in revised_bins_list:
        y_var= f"Percentage of road flooded more than {i}cm"
        for year in df.Year.unique():
            df_s = df.loc[df.Year == year]
            subtitle= y_var + f" in year {year}"
            fathom_barchart_infra_exposure(df_s,subtitle, y_var, x_var, cmap, maps, year, i)
            


        
def post_process_dir_cleaning(base_path, directories_to_remove):
    """
    List directories in the given path and remove the ones specified in the list.

    :param base_path: The path to search for directories.
    :param directories_to_remove: A list of directory names to remove.
    """
    # Get a list of directories in the base path
    all_directories = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    print(f"all_directories -- {all_directories}")
    # Iterate through all directories and remove those in the target list
    for directory in all_directories:
        if directory in directories_to_remove:
            dir_path = os.path.join(base_path, directory)
            try:
                shutil.rmtree(dir_path)  # Remove the directory and its contents
                print(f"Removed directory: {dir_path}")
            except Exception as e:
                print(f"Error removing directory {dir_path}: {e}")

            