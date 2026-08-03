# Soil Moisture Machine Learning Project
# 
# Trying to stimate soil moisture using Earth Engine remote sensing datasets
# and machine learning models.
#
# Note: This is expected to be a learning process and will include many "Ohhhhhhh thats why" moments


from src import gee_data, data_processing, model, visualization
import pandas as pd
import config

def main():

	gee_data.initialize_earth_engine()
	AOI = gee_data.load_aoi()
	model_nn = model.create_model()
	results = []
	
	all_actual = []
	all_predictions =[]

	for start_date, end_date in config.TIME_PERIODS:
        	print()
        	print("------------------------------------")
        	print(f"Training on {start_date} to {end_date}")
        	print("------------------------------------")
        
        	smap = gee_data.load_smap(AOI, start_date, end_date)

        	ndvi = gee_data.load_ndvi(AOI, start_date, end_date)

        	dem = gee_data.load_dem(AOI)
        	sentinel1 = gee_data.load_sentinel1(AOI, start_date, end_date)
        	sample_stack = gee_data.create_sample_stack(smap, ndvi, dem, sentinel1)

        	samples = gee_data.sample_stack(sample_stack, AOI)

        	print("Number of samples:", samples.size().getInfo())

        	dataframe = data_processing.feature_collection_to_dataframe(samples)

        	X_train, X_test, y_train, y_test = (
            	data_processing.prepare_training_data(dataframe)
        	)

        	model_nn = model.train_model(
            	model_nn,
            	X_train,
            	y_train
        	)

        	predictions, mae, rmse, r2 = model.evaluate_model(
            	model_nn,
            	X_test,
            	y_test
        	)
			
        	all_actual.extend(y_test)
        	all_predictions.extend(predictions)

	        results.append({
			"Start": start_date,
			"End": end_date,
			"Samples": len(dataframe),
			"MAE": mae, 
			"RMSE": rmse,
			"R2": r2
		})


	results_df = pd.DataFrame(results)
	print()
	print("Incremental Training Results")
	print(results_df)

	visualization.plot_predictions(
		all_actual,
		all_predictions
	)
	visualization.plot_training_progress(results_df)

if __name__ == "__main__":
	main()




















