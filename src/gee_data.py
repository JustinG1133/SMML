import ee
import config

def initialize_earth_engine():
	ee.Initialize(project = config.EE_PROJECT)
	print("Earth Engine initialized.")

def load_aoi():
	AOI = ee.FeatureCollection(config.AOI)
	print("AOI succussfully loaded from EE")

	return AOI


def load_smap(AOI, start_date, end_date):
	smap = (
	ee.ImageCollection(config.SMAP_DATASET)
	.filterDate(start_date, end_date)
	.filterBounds(AOI)
	)
	return smap


def load_ndvi(AOI, start_date, end_date):
	ndvi = (
		ee.ImageCollection(config.NDVI_DATASET)
		.filterDate(start_date, end_date)
		.filterBounds(AOI)
		)
	return ndvi

def load_dem(AOI):
	return (
		ee.ImageCollection(config.DEM_DATASET).mosaic()
		.clip(AOI)
		)

def create_sample_stack(smap, ndvi, dem):

    smap_mean = smap.mean().select(config.SMAP_BAND)

    ndvi_mean = ndvi.mean().select(config.NDVI_BAND)
    dem = dem.select(config.DEM_BAND)


    sample_stack = smap_mean.addBands(ndvi_mean)
    sample_stack = sample_stack.addBands(dem)

    return sample_stack

def sample_stack(sample_stack, AOI):
	samples = sample_stack.sample(
		region = AOI.geometry(),
		scale = config.SAMPLE_SCALE,
		numPixels = config.NUM_SAMPLES,
		seed = config.RANDOM_SEED,
		geometries = False
		)
	return samples



