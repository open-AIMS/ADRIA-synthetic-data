import os

global SYNTH_DATA_PACKAGE_DIR, SYNTH_DATA_DIR, ORIG_DATA_DIR
file_dir = os.path.dirname(os.path.abspath(__file__))
SYNTH_DATA_PACKAGE_DIR = file_dir[:-19]+"synthetic_data\\synthetic_data_packages\\"
SYNTH_DATA_DIR = file_dir[:-19]+"synthetic_data\\"
ORIG_DATA_DIR = file_dir[:-19]+"original_data\\"


def initialize_data_package(synth_data_stamp):
    synth_data_set_folder = SYNTH_DATA_PACKAGE_DIR+synth_data_stamp

    SITE_DATA_DIR = os.path.join(synth_data_set_folder, "site_data")
    CONN_DATA_DIR = os.path.join(synth_data_set_folder, "connectivity","2000")
    DHW_DATA_DIR = os.path.join(synth_data_set_folder, "DHWs")

    os.makedirs(SITE_DATA_DIR)
    os.makedirs(CONN_DATA_DIR)
    os.makedirs(DHW_DATA_DIR)

def create_synth_site_data_package_fp(time_stamp,N):
    return SYNTH_DATA_PACKAGE_DIR+time_stamp+'_numsamps_'+str(N)+'\\site_data\\site_data_anon_'+time_stamp+'_numsamps_'+str(N)+'.csv'

def create_synth_site_fp(time_stamp,N):
    return SYNTH_DATA_DIR+'site_data_'+time_stamp+'_numsamps_'+str(N)+'.csv'

def retrieve_synth_site_fp(synth_data_fn):
    return SYNTH_DATA_DIR+synth_data_fn

def retrieve_orig_site_data_fp(orig_data_package,file_type):
    return ORIG_DATA_DIR+orig_data_package+"\\site_data\\"+orig_data_package+file_type

def retrieve_orig_conn_fp(orig_data_package,year,num):
    return ORIG_DATA_DIR+orig_data_package+"\\connectivity\\"+year+"\\connect_matrix_"+year+"_"+num+".csv"

def retrieve_orig_cover_fp(orig_data_package):
    return ORIG_DATA_DIR+orig_data_package+'\\site_data\\coral_cover.nc'
 
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

def retrieve_synth_env_data_fp(synth_data_fn,layer):
    if layer=='dhw':
        file_loc = "\\DHWs\\"
    elif layer=='Ub':
        file_loc = "\\waves\\"

    return SYNTH_DATA_PACKAGE_DIR+synth_data_fn.split("\\")[1][9:-3]+file_loc+layer+"_"+synth_data_fn.split("\\")[1][9:-3]+"nc"
