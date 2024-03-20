import pandas as pd
import geopandas as gp
from shapely.geometry import Polygon

import sys

sys.path.append("..")

from src.data_processing.postprocess_functions import convert_to_geo

site_data = gp.read_file(
    "C:\\Users\\rcrocker\\Documents\\Github\\ADRIA.jl\\examples\\Test_domain\\site_data\\Test_domain.gpkg"
)
site_data_new = site_data.copy()
site_data_new["lat"] = site_data["geometry"].centroid.y
site_data_new["long"] = site_data["geometry"].centroid.x
site_data_new.drop(["geometry"], axis=1, inplace=True)

site_data_geo_new = convert_to_geo(site_data_new)
breakpoint()
site_data_geo_new["row_id"] = site_data_geo_new.index + 1
site_data_geo_new.reset_index(drop=True, inplace=True)
site_data_geo_new.set_index("row_id", inplace=True)

site_data_geo_new.to_file(
    "C:\\Users\\rcrocker\\Documents\\Github\\ADRIA.jl\\examples\\Test_domain\\site_data\\Test_domain_new.gpkg",
    driver="GPKG",
)
