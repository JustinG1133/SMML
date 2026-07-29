from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
import config

def create_model():

	model = MLPRegressor(	
		hidden_layer_sizes = config.HIDDEN_LAYER_SIZES,
		activation = config.ACTIVATION,
		max_iter = config.MAX_ITER,
		random_state = config.RANDOM_SEED,
		warm_start = True
		)

	return model

def train_model(model, X_train, y_train):

	model.fit(X_train, y_train)
	return model

def evaluate_model(model, X_test, y_test):
	
	predictions = model.predict(X_test)
	
	mae = mean_absolute_error(y_test, predictions)
	r2 = r2_score(y_test, predictions)
	rmse = root_mean_squared_error(y_test, predictions)

	print("Model Performance")
	print("-----------------")
	print(f"MAE : {mae:.6f}")
	print(f"R²  : {r2:.4f}")
	print(f"RMSE: {rmse:.6f}")
	
	return predictions, mae, rmse, r2
