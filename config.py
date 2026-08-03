# This is the config file for the ML model.
# It is meant to control the datasets: Dataset, Date/Time, AOI, sample size?

#Earth Engine
EE_PROJECT = "ee-jgray"

#Area of interest (AOI)
AOI = "projects/ee-jgray/assets/MS_AOI"

# Start / End date for the study period
TIME_PERIODS = [
	("2024-06-01", "2024-06-15"),
	("2024-06-16", "2024-06-30"),
	("2024-07-01", "2024-07-31"),
	("2024-08-01", "2024-08-31"),
	("2024-09-01", "2024-09-30"),
	("2024-10-01", "2024-10-31"),
	("2024-11-01", "2024-11-30"),
	("2024-12-01", "2024-12-31")
]
#Datasets
SMAP_DATASET = "NASA/SMAP/SPL3SMP_E/006"

NDVI_DATASET = "MODIS/061/MOD13Q1"
SENTINEL1_DATASET = "OPERA/RTC/L2_V1/S1"
DEM_DATASET = "JAXA/ALOS/AW3D30/V4_1"

# Bands to use
SMAP_BAND = [
    "soil_moisture_am",
    "vegetation_water_content_am"
]
SENTINEL1_BANDS = ["VV", "VH"]
NDVI_BAND = "NDVI"

DEM_BAND = "DSM"


#Sampleing?
SAMPLE_SCALE = 9000
NUM_SAMPLES = 10000
RANDOM_SEED = 42



#Machine Learning Configs?
INCREMENTAL_TRAINING = True
TEST_SIZE = 0.2

TARGET_BAND = "soil_moisture_am"

FEATURE_COLUMNS = [
    "DSM",
    "NDVI",
    "vegetation_water_content_am"
]

NDVI_SCALE_FACTOR = 10000

HIDDEN_LAYER_SIZES = (64, 32)
ACTIVATION = "relu"
MAX_ITER = 500





