#This file is used to visualize the model performace!

import matplotlib.pyplot as plt
import numpy as np

def plot_predictions(y_test, predictions):
	
	plt.figure(figsize = (6, 6))
	plt.scatter(y_test, predictions, alpha = 0.3, s = 12)
	

	y_test = np.array(y_test)
	predictions = np.array(predictions)
	
	min_value = min(y_test.min(), predictions.min())
	max_value = max(y_test.max(), predictions.max())
	

	plt.plot(
		[min_value, max_value],
		[min_value, max_value],
		"r--",
		linewidth = 2,
		)

	plt.xlabel("Actual Soil Moisture")
	plt.ylabel("Predicted Soil Moisture")
	plt.title("Predicted vs Actual SM Value")

	plt.grid(True)
	
	plt.tight_layout()
	plt.savefig("predicted_vs_actual.png")
	print("scatter plot generated as png")

def plot_training_progress(results_df):

	labels = results_df["Start"]
		
	# R2
	
	plt.figure(figsize=(8,5))
	plt.plot(
		labels,
		results_df["R2"],
		marker = "o"
	)
	plt.title("Model Performance (R²)")
	plt.xlabel("Time Period")
	plt.ylabel("R2")
	plt.grid(True)
	plt.tight_layout()
	plt.savefig("r2_progress.png")
	plt.close()

	# MAE
	plt.figure(figsize=(8,5))
	plt.plot(labels, results_df["MAE"], marker="o")
	plt.title("Model Performance (MAE)")
	plt.xlabel("Time Period")
	plt.ylabel("MAE")
	plt.grid(True)
	plt.tight_layout()
	plt.savefig("mae_progress.png")
	plt.close()

	# RMSE
	plt.figure(figsize=(8,5))
	plt.plot(labels, results_df["RMSE"], marker="o")
	plt.title("Model Performance (RMSE)")
	plt.xlabel("Time Period")
	plt.ylabel("RMSE")
	plt.grid(True)
	plt.tight_layout()
	plt.savefig("rmse_progress.png")
	plt.close()

	print("Training progress graphs saved.")
