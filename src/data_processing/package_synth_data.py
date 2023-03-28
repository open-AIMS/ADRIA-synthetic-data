import os
import json

global SYNTH_DATA_PACKAGE_DIR, SYNTH_DATA_DIR, ORIG_DATA_DIR
file_dir = os.path.dirname(os.path.abspath(__file__))
SYNTH_DATA_PACKAGE_DIR = file_dir[:-19]+"synthetic_data\\synthetic_data_packages\\"
SYNTH_DATA_DIR = file_dir[:-19]+"synthetic_data\\"
ORIG_DATA_DIR = file_dir[:-19]+"original_data\\"

def initialize_data_package(time_stamp):
    synth_data_set_folder = SYNTH_DATA_PACKAGE_DIR+'synth_'+time_stamp

    SITE_DATA_DIR = os.path.join(synth_data_set_folder, "site_data")
    CONN_DATA_DIR = os.path.join(synth_data_set_folder, "connectivity","2000")
    DHW_DATA_DIR = os.path.join(synth_data_set_folder, "DHWs")
    WAVE_DATA_DIR = os.path.join(synth_data_set_folder, "waves")

    os.makedirs(SITE_DATA_DIR)
    os.makedirs(CONN_DATA_DIR)
    os.makedirs(DHW_DATA_DIR)
    os.makedirs(WAVE_DATA_DIR)

def retrieve_orig_data_package_path(orig_data_package):
    return ORIG_DATA_DIR+orig_data_package

def retrieve_synth_data_package_path(synth_data_package):
    return SYNTH_DATA_PACKAGE_DIR+synth_data_package

def create_synth_site_data_package_fp(time_stamp):
    return SYNTH_DATA_PACKAGE_DIR+'synth_'+time_stamp+'\\site_data\\'+'synth_'+time_stamp+'.gpkg'

def create_synth_site_data_fp(time_stamp):
    return SYNTH_DATA_DIR+'site_data_'+'synth_'+time_stamp+'.csv'

def retrieve_synth_site_data_fp(synth_data_fn):
    return SYNTH_DATA_DIR+synth_data_fn

def retrieve_orig_site_data_fp(orig_data_package,file_type):
    return ORIG_DATA_DIR+orig_data_package+"\\site_data\\"+orig_data_package+file_type

def retrieve_orig_conn_fp(orig_data_package,year,num):
    return ORIG_DATA_DIR+orig_data_package+"\\connectivity\\"+year+"\\connect_matrix_"+year+"_"+num+".csv"

def retrieve_orig_cover_fp(orig_data_package):
    return ORIG_DATA_DIR+orig_data_package+'\\site_data\\coral_cover.nc'

def retrieve_synth_cover_fp(synth_data_package):
    return SYNTH_DATA_PACKAGE_DIR+synth_data_package+'\\site_data\\coral_cover.nc'
 
def retrieve_orig_env_fp(orig_data_package,rcp,layer):
    if layer=='dhw':
        file_loc = "\\DHWs\\"
        file = 'dhwRCP'+rcp+'.nc'
    elif layer=='Ub':
        file_loc = "\\waves\\"
        file = 'wave_RCP'+rcp+'.nc'
    else:
        ValueError("Unrecognised environmental data layer.")

    return ORIG_DATA_DIR+orig_data_package+file_loc+file

def retrieve_synth_env_data_fp(synth_data_fn,layer,rcp):
    if layer=='dhw':
        file_loc = "\\DHWs\\"
        file_name = "dhwRCP"+rcp
    elif layer=='Ub':
        file_loc = "\\waves\\"
        file_name = "wave_RCP"+rcp

    return SYNTH_DATA_PACKAGE_DIR+synth_data_fn[10:-4]+file_loc+file_name+".nc"

def retrieve_synth_conn_data_fp(synth_data_fn):
    return SYNTH_DATA_PACKAGE_DIR+synth_data_fn[10:-4]+"\\connectivity\\2000\\connectivity.csv"

def create_dp_jason(orig_data_package_path,synth_data_package_name):
    with open(orig_data_package_path+"\\datapackage.json","r+") as f:
        data = json.load(f)
    breakpoint()
    data['name'] = synth_data_package_name # <--- add `id` value.
    data['title'] = "Synthetic data package" # <--- add `id` value.
    data["description"] = "Data package synthesised from actual data package for ADRIA."
    data["resources"][0]['path'] = 'site_data\\'+synth_data_package_name+".gpkg"
    data["contributors"] = []
    synth_data_package_path = retrieve_synth_data_package_path(synth_data_package_name)

    with open(synth_data_package_path+"\\datapackage.json", 'w') as f:
        json.dump(data, f, indent=4)

