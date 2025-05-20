from flask import Flask, request, render_template, jsonify ,redirect, url_for,flash
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image
from bson.objectid import ObjectId
from pymongo import MongoClient
from werkzeug.utils import secure_filename
import base64
import matplotlib.pyplot as plt
from skimage import io
from io import BytesIO
import pandas as pd  # Import pandas for DataFrame operations
import os
import cv2
import uuid
from utilities import focal_tversky, tversky, tversky_loss  # Import custom functions
import pickle  # Import pickle for loading the model



last_uploaded_file = None

with open("model_pipeline.pkl", "rb") as f:
    model11 = pickle.load(f)# Load the classifier and segmentation models (modify paths accordingly)
classifier_model = tf.keras.models.load_model('classifier_model .keras')  # Replace with your model


segmentation_model = tf.keras.models.load_model(
    'ResUNet-segmentation.keras',  # Path to your saved model
    custom_objects={
        'focal_tversky': focal_tversky,
        'tversky': tversky
    }
)


# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'amijektomar' 
client = MongoClient('mongodb://localhost:27017/')
db = client.tumour


# Ensure the upload directory exists
upload_dir = 'static/uploads'
os.makedirs(upload_dir, exist_ok=True)


class_names = ['Healthy', 'Tumorous']

def classify_image(img_path, model):
    # Load image
    img = load_img(img_path, target_size=(256, 256))  # adjust size to your model's input
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # normalize if needed

    # Predict
    prediction = model.predict(img_array)
    
    # If model outputs probability
    if prediction.shape[-1] == 1:
        result = class_names[int(prediction[0][0] > 0.5)]
    else:
        result = class_names[np.argmax(prediction)]

    return result



# Load the second model for classifying tumor types
tumor_model_path = r"brain_tumor_classifier.h5"
tumor_model = tf.keras.models.load_model(tumor_model_path)

# Define tumor classes
TUMOR_CLASSES = ["Normal", "Glioma", "Meningioma", "Pituitary"]

@app.route("/check_type", methods=["POST"])
def check_type():
    try:
        data = request.get_json()
        image_base64 = data.get("image_url")

        if not image_base64:
            return jsonify({"error": "No image data provided"}), 400

        # Extract base64 string (remove prefix if present)
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]

        # Decode and save temporarily
        image_bytes = base64.b64decode(image_base64)
        filename = f"{uuid.uuid4().hex}.png"
        image_path = os.path.join("static/uploads", secure_filename(filename))

        with open(image_path, "wb") as f:
            f.write(image_bytes)

        # Preprocess
        img_size = 256  # Must match model input
        img = load_img(image_path, target_size=(img_size, img_size))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = tumor_model.predict(img_array)
        tumor_type = TUMOR_CLASSES[np.argmax(prediction)]

        return jsonify({"tumor_type": tumor_type})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


numeric_features = ['Age', 'Tumor_Size']  # add all your numeric feature names here


@app.route('/predict_details', methods=['POST'])
def predict_details():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"message": "No input data provided"}), 400

        # Convert to DataFrame
        input_df = pd.DataFrame([data])

        # Convert numeric fields to float
        for col in numeric_features:
            if col in input_df.columns:
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

        # Check for missing values after type conversion
        if input_df[numeric_features].isnull().any().any():
            return jsonify({"message": "Invalid input: numeric conversion failed."}), 400

        # Predict
        prediction = model11.predict(input_df)[0]

        return jsonify({"result": str(prediction)})

    except Exception as e:
        return jsonify({"message": f"Error occurred: {str(e)}"}), 500
    
    
    
# Load the trained model (make sure the path to your .keras file is correct)
model_path = r"C:\\Users\\minha\Desktop\\MajorProject\brain_tumor_detector1.keras"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
model = load_model(model_path)

# Set the target image size
img_size = 128  # The same size used for training

# Define the prediction function
def predict_tumour(image_path):
    try:
        # Load and process the image
        img = load_img(image_path, target_size=(img_size, img_size))
        img_array = img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Predict the class
        prediction = model.predict(img_array)[0][0]
        
        # Return the result
        if prediction > 0.5:
            return "Tumorous"
        else:
            return "Healthy"
    except Exception as e:
        return f"Error processing image: {e}"
    
    
def predict_segmentation(path, model_seg):
    # Step 2.1: Read the image
    img = io.imread(path)
    
    # Step 2.2: Create an empty array for the input (1, 256, 256, 3)
    X = np.empty((1, 256, 256, 3))
    
    # Step 2.3: Resize and preprocess the image
    img = cv2.resize(img, (256, 256))
    img = np.array(img, dtype=np.float64)
    
    # Standardize the image (zero mean and unit variance)
    img -= img.mean()
    img /= img.std()
    
    # Step 2.4: Reshape the image to (1, 256, 256, 3) for batch prediction
    X[0,] = img
    
    # Step 2.5: Predict using the model
    predict = model_seg.predict(X)

 
    return  predict[0]  # Return the predicted mask
    
    
    
# Load the trained model (make sure the path to your .keras file is correct)
model1_path = r"C:\\Users\\minha\Desktop\\MajorProject\\mri_classification_model.h5"
if not os.path.exists(model1_path):
    raise FileNotFoundError(f"Model file not found: {model1_path}")
model1 = load_model(model1_path)


img_size = 128  # The same size used for training


def predict_mri_image(image_path):
    try:
        # Load and process the image
        img = load_img(image_path, target_size=(img_size, img_size))
        img_array = img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Predict the class
        prediction = model1.predict(img_array)[0][0]
        
        # Return the result
        if prediction > 0.5:
            return "This is not a MRI image"
        else:
            return predict_tumour(image_path)
    except Exception as e:
        return f"Error processing image: {e}"
    
    

# Define the route to handle the home page
@app.route('/')
def index():
    return render_template('signup.html')  # HTML page where the user can upload images



@app.route('/register', methods=['POST'])
def register():
    if request.method == 'POST':
        u_name = request.form['username']
        _email = request.form['email']
        password = request.form['password']
        
        existing_user = db.users.find_one({'username': u_name})
        if existing_user:
            return 'Username already taken. Please choose a different username.', 400
        
        
        existing_email = db.users.find_one({'email': _email})
        if existing_email:
            return 'eamil already taken. Please choose a different email.', 400
        
        db.users.insert_one({'username': u_name, 'email': _email, 'password': password})
        return redirect(url_for('success'))

@app.route('/success')
def success():
    return render_template('index.html')

@app.route('/Login')
def login():
    return render_template('login.html')


@app.route('/reg')
def indx():
    return render_template('signup.html')







@app.route('/log_in', methods=['POST'])
def logr():
    if request.method == 'POST':
        u_name = request.form['username']
        password = request.form['password']
        
        
    
    
    existing_u = db.users.find_one( { 'username': u_name }, {'password': password })
    
    if existing_u:
       
            if u_name =="Minhaz":
                return redirect(url_for('admin'))
            else:
                return redirect(url_for('success'))
        
    else:
        return'incorrect password or username'
    
def tif_to_base64_png(tif_path):
    image = Image.open(tif_path).convert("RGB")  # TIFFs are often grayscale or 16-bit
    buffer = BytesIO()
    image.save(buffer, format="PNG")  # Convert to PNG
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')    


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the image
    image_path = os.path.join('static/uploads', secure_filename(file.filename))
    file.save(image_path)

    # Classification
    result = classify_image(image_path, classifier_model)



    image_base64 = tif_to_base64_png(image_path)
    # Only perform segmentation if the result is Tumorous
    if result == "Tumorous":
        prediction_mask = predict_segmentation(image_path, segmentation_model)
        if prediction_mask is not None:
            binary_mask = (prediction_mask > 0.5).astype(np.uint8)

            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            ax[0].imshow(io.imread(image_path))
            ax[0].set_title("Original Image")
            ax[0].axis("off")

            ax[1].imshow(binary_mask, cmap='gray')
            ax[1].set_title("Segmentation Mask")
            ax[1].axis("off")

            img_stream = BytesIO()
            plt.savefig(img_stream, format='png')
            plt.close(fig)
            img_stream.seek(0)

            segmented_image_base64 = base64.b64encode(img_stream.getvalue()).decode('utf-8')

            return jsonify({
                'result': result,
                'segmented_image_base64': segmented_image_base64,
                'image_base64': image_base64
            })

    # If Healthy, return only the original image
    return jsonify({
        'result': result,
        'image_base64': image_base64
    })

if __name__ == '__main__':
    app.run(debug=True)

