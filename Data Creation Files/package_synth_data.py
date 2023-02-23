import os

def initialize_data_package(synth_data_stamp):
    synth_data_set_folder = "Synthetic Data\\Synthetic Data Packages\\"+synth_data_stamp

    SITE_DATA_DIR = os.path.join(synth_data_set_folder, "site_data")
    CONN_DATA_DIR = os.path.join(synth_data_set_folder, "connectivity","2000")
    DHW_DATA_DIR = os.path.join(synth_data_set_folder, "DHWs")

    os.makedirs(SITE_DATA_DIR)
    os.makedirs(CONN_DATA_DIR)
    os.makedirs(DHW_DATA_DIR)

def retrieve_synth_site_data_fp(synth_site_data_fn):
    return "Synthetic Data\\"+synth_site_data_fn

def retrieve_orig_site_data_fp(orig_data_package):
    return "Original Data\\"+orig_data_package+"\\site_data\\"+orig_data_package+".csv"

def retrieve_orig_conn_fp(orig_data_package,year,num):
    return "Original Data\\"+orig_data_package+"\\connectivity\\"+year+"\\connect_matrix_"+year+"_"+num+".csv"

def retrieve_orig_cover_fp(orig_data_package):
    return 'Original Data\\'+orig_data_package+'\\site_data\\coral_cover.nc'
 
def retrieve_orig_env_fp(orig_data_package,rcp,layer):
    if layer=='dhw':
        file_loc = "\\DHWs\\"
        file = 'dhwRCP'+rcp+'.nc'
    elif layer=='Ub':
        file_loc = "\\waves\\"
        file = 'wave_RCP'+rcp+'.nc'
    else:
        ValueError("Unrecognised environmental data layer.")

    return "Original Data\\"+orig_data_package+file_loc+file

def retrieve_synth_env_data_fp(synth_data_fn,layer):
    if layer=='dhw':
        file_loc = "\\DHWs\\"
    elif layer=='Ub':
        file_loc = "\\waves\\"

    return"Synthetic Data\\Synthetic Data Packages\\"+synth_data_fn.split("\\")[1][9:-3]+file_loc+layer+"_"+synth_data_fn.split("\\")[1][9:-3]+"nc"
