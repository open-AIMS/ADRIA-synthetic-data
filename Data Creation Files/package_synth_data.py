import os

def initialize_data_package(synth_data_stamp):
    synth_data_set_folder = "Synthetic Data\\Synthetic Data Packages\\"+synth_data_stamp

    SITE_DATA_DIR = os.path.join(synth_data_set_folder, "site_data")
    CONN_DATA_DIR = os.path.join(synth_data_set_folder, "connectivity","2000")
    DHW_DATA_DIR = os.path.join(synth_data_set_folder, "DHWs")

    os.makedirs(SITE_DATA_DIR)
    os.makedirs(CONN_DATA_DIR)
    os.makedirs(DHW_DATA_DIR)

