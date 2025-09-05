# Import necessary libraries
import tensorflow as tf
from keras.models import load_model            # For loading the trained model
from PIL import Image, ImageOps                # For image loading and preprocessing
import numpy as np                             # For numerical operations and arrays

# ------------------------------------------------------------
# Load the trained model
# Note: compile=False is used because we are not re-training the model
# IMPORTANT: Use raw string (r"...") to avoid issues with backslashes in Windows paths
model = load_model(
    r"# copy and past the correct path here\keras_model.h5",
    compile=False
)

# ------------------------------------------------------------
# Load the class labels (e.g., "Cat", "Dog")
# The labels.txt file usually has one class per line
with open(r"# copy and past the correct path here\labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]  # Remove newline characters

# ------------------------------------------------------------
# Define a function to preprocess the image and make prediction
def predict_image(image_path):
    # Create an empty array with the shape the model expects (1 image, 224x224, 3 channels)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Open the image and convert to RGB (in case it's grayscale or CMYK)
    image = Image.open(image_path).convert("RGB")

    # Resize and crop the image to fit the model's expected input size
    image = ImageOps.fit(image, (224, 224), Image.LANCZOS)

    # Convert the image to a NumPy array
    image_array = np.asarray(image)

    # Normalize the image to range [-1, 1] as expected by Teachable Machine models
    normalized = (image_array.astype(np.float32) / 127.0) - 1

    # Put the image data into the input array
    data[0] = normalized

    # --------------------------------------------------------
    # Make a prediction
    prediction = model.predict(data)

    # Find the index of the class with the highest probability
    index = np.argmax(prediction[0])

    # Return:
    # - the predicted class name
    # - its confidence score
    # - all class probabilities (optional)
    return class_names[index], float(prediction[0][index]), prediction[0]

# ------------------------------------------------------------
# Example usage of the function

# Path to the image you want to classify
image_path = r"# copy and past the correct path here\test_image.jpg"

# Get prediction
class_name, confidence, all_preds = predict_image(image_path)

# Print results
print("Predicted class:", class_name)
print("Confidence:", f"{confidence * 100:.2f}%")
print("All scores:", all_preds)  # Optional: useful for debugging or multi-class models
