from flask import Flask, request, render_template, jsonify ,redirect, url_for,flash
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image
from bson.objectid import ObjectId
from pymongo import MongoClient


import os


last_uploaded_file = None



# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'amijektomar' 
client = MongoClient('mongodb://localhost:27017/')
db = client.tumour


# Ensure the upload directory exists
upload_dir = 'static/uploads'
os.makedirs(upload_dir, exist_ok=True)




# Load the second model for classifying tumor types
tumor_model_path = r"C:\\Users\\minha\Desktop\\MajorProject\brain_tumor_classifier.h5"
tumor_model = tf.keras.models.load_model(tumor_model_path)

# Define tumor classes
TUMOR_CLASSES = ["Normal", "Glioma", "Meningioma", "Pituitary", "Metastatic"]

@app.route("/check_type", methods=["POST"])
def check_type():
    global last_uploaded_file
   
    image_path =last_uploaded_file

    if not image_path:
        return jsonify({"error": "No image provided"}), 400

    try:
        # Load and preprocess the image
        img_size = 128  # Must match model input size
        img = load_img(image_path, target_size=(img_size, img_size))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Predict the tumor type
        prediction = tumor_model.predict(img_array)
        tumor_type = TUMOR_CLASSES[np.argmax(prediction)]  # Get the predicted class

        return jsonify({"tumor_type": tumor_type})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

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


# Define the route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    
    global last_uploaded_file  # Use the global variable
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Save the uploaded file to a temporary directory
    file_path = os.path.join('static/uploads', file.filename)
    file.save(file_path)
    last_uploaded_file = file_path 
    
    image_url = f'/static/uploads/{file.filename}'
    
    # Predict the class of the image
    result = predict_mri_image(file_path)
    return jsonify({'result': result, 'image_url': image_url})

    
if __name__ == '__main__':
    app.run(debug=True)
