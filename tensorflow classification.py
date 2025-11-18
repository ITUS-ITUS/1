from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import keras
from keras import layers
iris = load_iris()
X, y = iris.data, iris.target
y = to_categorical(y)  # one-hot encode labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = keras.Sequential([
    layers.Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    layers.Dense(16, activation="relu"),
    layers.Dense(3, activation="softmax")  # 3 classes
])

# Compile
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train
model.fit(X_train, y_train, validation_split=0.2, epochs=30, batch_size=32)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", acc)
