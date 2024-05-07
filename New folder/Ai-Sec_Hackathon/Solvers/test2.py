import numpy as np
import tensorflow as tf
from keras.models import load_model

# from tensorflow.keras.models import load_model

# Load data from the .npz file
real = np.load("Solvers/real.npz")
fake = np.load("Solvers/fake.npz")
X_real = real["x"]
y_real = np.ones(750)
X_fake = fake["x"]
y_fake = np.zeros(750)
print(y_fake[0])


def image_preprocessing(image):
    image[np.isinf(image)] = float(0)
    image = np.expand_dims(image, axis=0)
    return image


fake_test = image_preprocessing(X_fake[200])
fake_test1 = image_preprocessing(X_fake[55])
real_test = image_preprocessing(X_real[100])

model = load_model("Solvers/best_model_lastone.h5")

ls = [fake_test, fake_test1, real_test]

print("============Selecting Channel==========")

for i, img in enumerate(ls):
    pred = (model.predict(img) >= 0.5).astype(int).flatten()[0]

    if pred == 1:
        print(i + 1)

print("No valid channel selected")
print("=======================================")
print(int(-1))
