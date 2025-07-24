from flask import Flask, request, render_template
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import os

app = Flask(__name__)
model = load_model('skin_cancer_cnn.h5')

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def predict_skin_cancer(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    class_label = "Malignant" if prediction > 0.5 else "Benign"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    return class_label, confidence

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file uploaded', 400
        file = request.files['file']
        if file.filename == '':
            return 'No selected file', 400
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            class_label, confidence = predict_skin_cancer(filepath)
            return render_template('index.html', filename=file.filename, 
                                   prediction=class_label, confidence=confidence)

    return render_template('index.html')

@app.route('/display/<filename>')
def display_image(filename):
    return f'<img src="/static/uploads/{filename}" width="300">'

if __name__ == "__main__":
    app.run(debug=True)
