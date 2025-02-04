import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import main2
import torch
import torchvision.transforms as transforms
from PIL import Image

app = Flask(__name__)

# Folder to store uploaded files
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for Home Page
@app.route('/')
def home():
    return render_template('home.html')

# Route for Upload Form Page
@app.route('/upload', methods=['GET', 'POST'])
def upload_page():
    if request.method == 'POST':
        name = request.form.get('name')
        aadhar = request.files.get('aadhar')
        pan = request.files.get('pan')

        error_message = None  # Variable to hold error message
        aadhar_path = None
        pan_path = None

        # Ensure both Aadhar and PAN are uploaded
        if not aadhar or not pan:
            return render_template('upload_form.html', error="Both Aadhar and PAN card are required.")

        # Save Aadhar card
        if aadhar and allowed_file(aadhar.filename):
            aadhar_path = os.path.join(app.config['UPLOAD_FOLDER'], aadhar.filename)
            aadhar.save(aadhar_path)
        else:
            return render_template('upload_form.html', error="Invalid Aadhar card format.")

        # Save PAN card
        if pan and allowed_file(pan.filename):
            pan_path = os.path.join(app.config['UPLOAD_FOLDER'], pan.filename)
            pan.save(pan_path)
        else:
            return render_template('upload_form.html', error="Invalid PAN card format.")

        # Get model predictions
        aadhar_prediction = main2.predict(aadhar_path)
        pan_prediction = main2.predict(pan_path)

        # Validate both predictions
        if aadhar_prediction != "Aadhar_Card":
            os.remove(aadhar_path)  # Delete incorrect file
            os.remove(pan_path)
            return render_template('upload_form.html', error="Uploaded Aadhar card is incorrect.")
        
        if pan_prediction != "Pan_Card":
            os.remove(aadhar_path)
            os.remove(pan_path)  # Delete incorrect file
            return render_template('upload_form.html', error="Uploaded PAN card is incorrect.")

        # Redirect to success page if valid
        return redirect(url_for('success_page'))

    return render_template('upload_form.html')

# Route to Display Success Page
@app.route('/success')
def success_page():
    return render_template('tick.html')

# Route to Serve Uploaded Files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
