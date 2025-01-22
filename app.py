import os
import numpy as np
import cv2
from joblib import load
import pandas as pd
import tensorflow as tf
from flask import Flask, render_template, request, redirect

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path='models/feature_extractor.tflite')

# Allocate tensors for the interpreter
interpreter.allocate_tensors()

# Load the Random Forest Classifier and label map
rfc = load('models/artifact_classifier.pkl')
label_map = load('models/labels.pkl')

# Load the artifact details
artifact_info = pd.read_csv('models/met_artifacts_combined_descriptions.csv')

# Image size used for prediction
image_size = (128, 128)

def predict_artifact(image_path):
    # Preprocess input image
    img = cv2.imread(image_path)
    img = cv2.resize(img, image_size)
    img = img / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)

    # Get input and output details for the TensorFlow Lite interpreter
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set input tensor for the model
    interpreter.set_tensor(input_details[0]['index'], img)

    # Run inference
    interpreter.invoke()

    # Extract features from the output tensor
    features = interpreter.get_tensor(output_details[0]['index'])

    # Predict using Random Forest Classifier
    predicted_class = rfc.predict(features)

    # Get the object number from label_map
    predicted_object_number = label_map[predicted_class[0]]

    # Get artifact details
    artifact_details = artifact_info[artifact_info['Object Number'] == predicted_object_number].iloc[0]

    artifact_name = artifact_details['Artifact Name']
    artifact_date = artifact_details['Date']
    artifact_culture = artifact_details['Culture/Region']
    artifact_material = artifact_details['Material']
    artifact_dimensions = artifact_details['Dimensions']
    artifact_category = artifact_details['Category/Type']
    artifact_description = artifact_details['Description']

    return predicted_object_number, artifact_name, artifact_date, artifact_culture, artifact_material, artifact_dimensions, artifact_category, artifact_description


# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file:
        # Save the uploaded file
        file_path = os.path.join('static/uploads', file.filename)
        file.save(file_path)
        
        # Call predict_artifact function
        artifact_details = predict_artifact(file_path)
        
        # Pass artifact details to the template for display
        return render_template('index.html', artifact_details=artifact_details)

if __name__ == '__main__':
    app.run(debug=True)