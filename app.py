from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import subprocess
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/input'
OUTPUT_FOLDER = 'static/output/all_images'
ALLOWED_EXTENSIONS = {'exe'}
MODEL_PATH = 'CNN.h5'  # Path to your trained model

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Load the pre-trained CNN model
model = load_model(MODEL_PATH)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def empty_folders():
    # Remove all files in the upload and output folders
    for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER']]:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove file
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)  # Remove folder (only if empty)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        return upload_file()
    return render_template('index.html')

@app.route('/static/output', methods=['POST'])
def upload_file():
    empty_folders()  # Clear the input and output folders before uploading a new file

    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Run the mallook.py script
        run_mallook(file_path)
        
        # Run predictions on the output images
        prediction_results, probabilities, rf_predictions = make_predictions_on_images()

        # Calculate average probability for CNN
        average_probability = np.mean(probabilities) if probabilities else 0

        # Check if the images were created successfully
        output_images = os.listdir(app.config['OUTPUT_FOLDER'])
        output_image_paths = [os.path.join('static/output/all_images', img) for img in output_images] if output_images else []

        # Ensure all lists have the same length
        min_length = min(len(output_image_paths), len(prediction_results), len(rf_predictions))

        # Trim the lists to the minimum length to avoid mismatched zipping
        output_image_paths = output_image_paths[:min_length]
        prediction_results = prediction_results[:min_length]
        rf_predictions = rf_predictions[:min_length]

        # Zip image paths and predictions before passing them to the template
        combined_results = zip(output_image_paths, prediction_results, rf_predictions)

        # Pass the combined results, average, and probabilities to the template
        return render_template('result.html', combined_results=combined_results, average_probability=average_probability, probabilities=probabilities)
    
    return redirect(url_for('index'))


def run_mallook(input_file_path):
    command = ['python', 'mallook.py', input_file_path]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")

def make_predictions_on_images():
    # Preprocess the images in the output folder and make predictions
    image_folder_path = app.config['OUTPUT_FOLDER']
    preprocessed_images = []
    image_names = []

    for image_name in os.listdir(image_folder_path):
        image_path = os.path.join(image_folder_path, image_name)
        image = cv2.imread(image_path)
        
        if image is not None:
            # Resize the image to the expected input size of the model
            image = cv2.resize(image, (224, 224))
            image = image.astype('float32') / 255.0  # Scale pixel values to [0, 1]
            
            # Add the preprocessed image to the list
            preprocessed_images.append(image)
            image_names.append(image_name)
        else:
            print(f"Could not load image: {image_name}")

    # Convert the list of images to a numpy array
    if preprocessed_images:
        preprocessed_images = np.array(preprocessed_images)
        # Predict for all images in the batch
        predictions = model.predict(preprocessed_images)

        # Interpret and store the results
        results = []
        probabilities = []

        for image_name, prediction in zip(image_names, predictions):
            prob = float(prediction[0])  # Convert numpy.float32 to standard Python float
            probabilities.append(prob)  # Collect the probability
            if prob > 0.5:
                results.append(f"{image_name}: Malicious (Probability: {prob:.2f})")
            else:
                results.append(f"{image_name}: Benign (Probability: {prob:.2f})")

        return results, probabilities
    else:
        return ["No images found for predictions."], []

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    
    app.run(debug=True)
