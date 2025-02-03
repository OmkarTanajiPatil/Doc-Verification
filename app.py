import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory

app = Flask(__name__)

# Folder to store uploaded files
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}

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

        if aadhar and allowed_file(aadhar.filename):
            aadhar_path = os.path.join(app.config['UPLOAD_FOLDER'], aadhar.filename)
            aadhar.save(aadhar_path)

        if pan and allowed_file(pan.filename):
            pan_path = os.path.join(app.config['UPLOAD_FOLDER'], pan.filename)
            pan.save(pan_path)

        # Redirect to the success page (tick.html) after upload
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
