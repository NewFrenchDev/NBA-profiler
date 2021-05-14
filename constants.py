import os

ROOT = os.path.dirname(__file__)
DATE_FORMAT = '%m/%d/%Y'
TEAMS_DATA_PATH = os.path.join(ROOT, 'data', 'teams') 
TEAMS_BOXSCORE_PATH = os.path.join(ROOT, TEAMS_DATA_PATH, 'teams-simplified-boxscore.csv') 
FILE_GZ_TO_DECOMPRESS = os.path.join(ROOT, TEAMS_DATA_PATH, 'teams-simplified-boxscore.tar.gz') 
MODELS_FILES = {
    'Logistic Regression': 'logistic_regression_model',
    'Decision Tree': "decision_tree_model",
    'Random Forest': "random_forest_model",
    'Gradient Boosting': "gradient_boosting_model",
    'K Nearest Neighbors': "knn_model",
    'AdaBoost': "adaboost_model",
    'Artificial Neural Network': "artificial_neural_network_model"
}
DATAFRAME_COLUMNS_TYPE = {
    'Points': 'int16',
    'Field Goal Made': 'int16',	
    'Field Goal Attempt': 'int16',	
    'Percent of Field Goal Made': 'float16',
    '3 Points Made':  'int8',
    '3 Points Attempt': 'int8',	
    'Percent of 3 Points Made': 'float16',
    'Free Throw Made': 'int8',
    'Free Throw Attempt': 'int8',
    'Percent of Free Throw Made': 'float16',
    'Offensive Rebounding': 'int8',
    'Defensive Rebounding': 'int8',	
    'Rebounding': 'int8',	
    'Assist': 'int8',	
    'Steal': 'int8',
    'Block': 'int8',
    'Turnover': 'int8',
    'Personal Fouls': 'int8',
    'ASTRatio': 'float16',	
    'Offensive rebounding percentage': 'float16',
    'Defensive rebounding percentage': 'float16',
    'Rebounding percentage': 'float16',	
    'Turnover percentage': 'float16',
    'Effective Field Goal Percentage': 'float16',
    'True Shooting Percentage': 'float16',	
    'Player Impact Estimate': 'float16',
    'Percent of Points (2-Point Field Goals)': 'float16',
    'Percent of Points (2-Point Field Goals: Mid Range)': 'float16',
    'Percent of Points (3-Point Field Goals)': 'float16',
    'Percent of Points (Free Throws)': 'float16',
    'Year': 'int16',	
    'Opponent Effective Field Goal Percentage': 'float16',
    'Opponent Turnover percentage': 'float16',
    'Opponent Offensive rebounding percentage': 'float16',
    'Opponent Percent of Points (Free Throws)': 'float16'
}

CODE_BY_MODEL = {

    "Logistic Regression": """
from sklearn.linear_model import LogisticRegression

#Create model
model = LogisticRegression()

#Train model
model.fit(X_train, Y_train)

#Save model
with open('logistic_regression_model', 'wb') as f:
    pickle.dump(model, f)

#Evaluate model
print("Accuracy on training set : ", accuracy_score(Y_train, Y_train_pred))
print("Accuracy on test set : ", accuracy_score(Y_test, Y_test_pred))
    """,

    "Decision Tree":"""
from sklearn.tree import DecisionTreeClassifier

#Create model
model = DecisionTreeClassifier(min_samples_leaf=5 , random_state=42)

#Train model
model.fit(X_train, Y_train)

#Save model
with open('decision_tree_model', 'wb') as f:
  pickle.dump(model, f)

# Evaluation du modèle 
print("Decision Tree Train score : {}".format(model.score(X_train, Y_train)))
print("Decision Tree Test score : {}".format(model.score(X_test, Y_test)))
    """,

    "Random Forest": """
from sklearn.ensemble import RandomForestClassifier

#Create model
model = RandomForestClassifier(n_estimators=15, random_state = 42)

#Train model
model.fit(X_train, Y_train)

#Save model
with open('random_forest_model', 'wb') as f:
  pickle.dump(model, f)

#Evaluate model
print(" Random forest Train score : {}".format(model.score(X_train, Y_train)))
print(" Random forest Test score : {}".format(model.score(X_test, Y_test)))
    """,

    "Gradient Boosting": """
from sklearn.ensemble import GradientBoostingClassifier

#Create model
model = GradientBoostingClassifier(loss="deviance",
                                   learning_rate=0.2,
                                   max_depth=5,
                                   max_features="sqrt",
                                   subsample=0.95,
                                   n_estimators=200)

#Train model                                   
model.fit(X_train, Y_train)

#Save model
with open('gradient_boosting_model', 'wb') as f:
  pickle.dump(model, f)

# Evaluate model 
print(" Gradient Boosting Train score : {}".format(model.score(X_train, Y_train)))
print(" Gradient Boosting Test score : {}".format(model.score(X_test, Y_test)))
    """,

    "K Nearest Neighbors": """
from sklearn.neighbors import KNeighborsClassifier

#Create model
model = KNeighborsClassifier(n_neighbors=100)

#Train model
model.fit(X_train, Y_train)

#Save model
with open('knn_model', 'wb') as f:
  pickle.dump(model, f)

#Evaluate model
print("KNN Train score : {}".format(model.score(X_train, Y_train)))
print("KNN Test score : {}".format(model.score(X_test, Y_test)))
    """,

    "AdaBoost": """
from sklearn.ensemble import AdaBoostClassifier

#Create a weak learner with Decision Tree
weak_learner = DecisionTreeClassifier(max_depth=1)

#Create model
model = AdaBoostClassifier(base_estimator=weak_learner, n_estimators=100, learning_rate=0.7)

#Train model
model.fit(X_train, Y_train)

#Save model
with open('adaboost_model', 'wb') as f:
  pickle.dump(model_ada_tree, f)

# Evaluate model
print("AdaBoost Train score : {}".format(model.score(X_train, Y_train)))
print("AdaBoost Test score : {}".format(model.score(X_test, Y_test)))
    """,

    "Artificial Neural Network": """
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from sklearn.preprocessing import  MinMaxScaler

#Create model
model = Sequential([
    Dense(units=16, input_shape=(8,), activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=2, activation='softmax')
])

#Check model summary
model.summary()

#Compile the model with the optimizer
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Split the train set from the test set
X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size=0.1, random_state=42)

#Convert train test to a numpy array (all features are numeric)
X_train = np.array(X_train)

#Convert W and L to 0 and 1 
Y_train = labelencoder.fit_transform(Y_train)

#Normalize all features between 0 and 1 
scaler = MinMaxScaler(feature_range=(0,1))
scaled_X_train = scaler.fit_transform(X_train)

#Train model
model.fit(x=scaled_X_train, y=Y_train, validation_split=0.2, batch_size=10, epochs=40, shuffle=True, verbose=2)

#Get some prediction with test set
scaled_X_test = scaler.fit_transform(X_test)
predictions = model.predict(x=scaled_X_test, batch_size=10, verbose=0)

#Get the index with best probability (0 = L, 1 = W)
rounded_predictions = np.argmax(predictions, axis=-1)

#Show the predictions
number_good_prediction = 0
number_test_case = 2000
for predict, round_predict, result in zip(predictions[:number_test_case], rounded_predictions[:number_test_case], Y_test2[:number_test_case]):
  if (round_predict == 1 and result == "W") or (round_predict == 0 and result == "L"):
    indication = "OK"
    number_good_prediction += 1 
  else:
    indication = "NOK"
    
  print(predict, round_predict, result, indication)

print("Number of good predictions: ", number_good_prediction)
print("Number of failed predictions: ", number_test_case - number_good_prediction)

    """
}