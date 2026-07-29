import pandas as pd
from sklearn.preprocessing import StandardScaler
import config

from sklearn.model_selection import train_test_split

def feature_collection_to_dataframe(samples):

    feature_list = samples.getInfo()["features"]
    data = [feature["properties"] for feature in feature_list]
    dataframe = pd.DataFrame(data)

    return dataframe


def prepare_training_data(dataframe):
	dataframe["NDVI"] = dataframe["NDVI"] / config.NDVI_SCALE_FACTOR
	
	X = dataframe[config.FEATURE_COLUMNS]

	y = dataframe[config.TARGET_BAND]
	
	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size = config.TEST_SIZE,
		random_state = config.RANDOM_SEED
	)

	scalar = StandardScaler()
	X_train = scalar.fit_transform(X_train)
	X_test = scalar.transform(X_test)

	return X_train, X_test, y_train, y_test



