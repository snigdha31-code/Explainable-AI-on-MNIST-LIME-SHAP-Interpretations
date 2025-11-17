
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input

# Explainability libs
from lime import lime_image
from skimage.segmentation import mark_boundaries
import shap

# Load MNIST
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype("float32")/255.0
x_test = x_test.astype("float32")/255.0

x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

# Build CNN
def build_model():
    model = Sequential([
        Input(shape=(28,28,1)),
        Conv2D(32,(3,3),activation='relu'),
        MaxPooling2D((2,2)),
        Conv2D(64,(3,3),activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128,activation='relu'),
        Dropout(0.3),
        Dense(10,activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = build_model()
model.fit(x_train,y_train,epochs=3,batch_size=128,validation_split=0.1)

# Pick test image

index = 20
test_img = x_test[index:index+1]          # shape (1,28,28,1)
raw_img = (test_img[0].squeeze()*255).astype(np.uint8)

pred = model.predict(test_img)
pred_label = int(np.argmax(pred))

print("Prediction:", pred_label, "True:", int(y_test[index]))

 
# LIME
def predict_fn_for_lime(images):
    images = np.array(images).astype("float32")

    if images.ndim == 3:
        images = images[np.newaxis, ...]

    if images.shape[-1] == 3:
        images = images.mean(axis=-1, keepdims=True)

    images = tf.image.resize(images, (28,28)).numpy()

    if images.max() > 1:
        images = images / 255.0

    return model.predict(images)

rgb_img = np.repeat(test_img[0], 3, axis=2)

lime_explainer = lime_image.LimeImageExplainer()
lime_exp = lime_explainer.explain_instance(
    image=rgb_img,
    classifier_fn=predict_fn_for_lime,
    top_labels=1,
    hide_color=0,
    num_samples=1000
)

top_label = lime_exp.top_labels[0]
lime_img, lime_mask = lime_exp.get_image_and_mask(
    top_label,
    positive_only=True,
    num_features=8,
    hide_rest=False
)


# SHAP (ImageExplainer)
print("Running SHAP ImageExplainer...")

masker = shap.maskers.Image("inpaint_telea", (28,28,1))
explainer = shap.Explainer(model, masker)

shap_values = explainer(test_img, max_evals=200)


# VISUALIZATION

plt.figure(figsize=(15,5))

# Original
plt.subplot(1,3,1)
plt.title("Original")
plt.imshow(raw_img, cmap="gray")
plt.axis("off")

# LIME
plt.subplot(1,3,2)
plt.title("LIME")
lime_vis = mark_boundaries(lime_img/255.0, lime_mask)
plt.imshow(lime_vis)
plt.axis("off")

# SHAP Heatmap
plt.subplot(1,3,3)
plt.title("SHAP Heatmap")
plt.imshow(shap_values.values[0].sum(axis=-1), cmap="RdBu")
plt.colorbar()
plt.axis("off")

plt.tight_layout()
plt.show()
