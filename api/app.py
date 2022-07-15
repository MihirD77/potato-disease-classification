from flask import Flask, render_template, request
import os
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename


app = Flask(__name__)

MODEL = tf.keras.models.load_model("../saved_models/1")
CLASS_NAMES = ["POTATO EARLY BLIGHT", "POTATO LATE BLIGHT", "POTATO HEALTHY"]


def image_to_array(data) -> np.ndarray:
    image_as_array = np.array(Image.open(BytesIO(data)))
    return image_as_array


@app.route('/', methods=['GET'])
def hello():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    f = request.files['file']
    base_path = os.path.dirname(__file__)
    os.path.join(base_path, 'uploads', secure_filename(f.filename))

    in_bytes = f.read()
    image = image_to_array(in_bytes)
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    print('Predictions: ' + predicted_class)
    print('Confidence: ' + str(confidence))

    return predicted_class


if __name__ == "__main__":
    app.run(port=3000, debug=True)
