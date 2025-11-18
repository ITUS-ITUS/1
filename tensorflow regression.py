import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
data = load_diabetes()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = keras.Sequential([
    layers.Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    layers.Dense(64, activation="relu"),
    layers.Dense(1)  # regression output → single neuron, linear activation
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
history = model.fit(X_train, y_train, 
                    validation_split=0.2, 
                    epochs=70, 
                    batch_size=32, 
                    verbose=1)
y_pred = model.predict(X_test).flatten()
print("R² Score:", r2_score(y_test, y_pred))
