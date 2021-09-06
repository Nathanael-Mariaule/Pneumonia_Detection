import os
from app import app
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from pneumonia_prediction import predict
import tensorflow as tf




ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
os.environ["CUDA_VISIBLE_DEVICES"]="-1"  
physical_devices = tf.config.list_physical_devices('CPU') 
mariaunet = tf.keras.models.load_model('mariaunet')


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
    """
    	display upload page
    """
    return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_image():
	"""
    	display image with the tensorflow model prediction
	"""
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	elif file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'xray.jpg')
		if os.path.exists(filepath):
			os.remove(filepath)
		file.save(filepath)
		flash('Image successfully uploaded and displayed below')
		predict(filepath, mariaunet)
		return render_template('upload.html', filename='xray.jpg')
	else:
		flash('Allowed image types are -> png, jpg, jpeg')
		return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
	"""
        display image
    """
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run()
