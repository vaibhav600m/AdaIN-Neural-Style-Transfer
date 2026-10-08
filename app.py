import sys
import os
from pathlib import Path

# Local project imports path setup
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from flask import Flask, render_template, request, send_from_directory
from flask_wtf import FlaskForm
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField
from PIL import Image
from torchvision import transforms

# Local AdaIN modules
from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class UploadForm(FlaskForm):
    content = FileField('Content Image')
    style = FileField('Style Image')
    content_path = HiddenField()
    style_path = HiddenField()
    alpha = FloatField('Alpha', default=1.0)
    submit = SubmitField('Transfer Style')

# MacBook MPS / CUDA / CPU Device Resolution
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Flask App running on device: {device}")

# Model Loaders with Local Paths
vgg_weights = 'weights/vgg_normalised.pth'
decoder_weights = 'weights/decoder_240.pth' # Aapka trained decoder checkpoint

encoder = VGGEncoder(vgg_weights).to(device)
decoder = Decoder().to(device)
decoder.load_state_dict(torch.load(decoder_weights, map_location=device))

encoder.eval()
decoder.eval()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def style_transfer(content_image, style_image, encoder, decoder, alpha, device):
    content_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])

    style_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])

    content_tensor = content_transform(content_image).unsqueeze(0).to(device)
    style_tensor = style_transform(style_image).unsqueeze(0).to(device)

    with torch.no_grad():
        content_feats = encoder(content_tensor, is_test=True)
        style_feats = encoder(style_tensor, is_test=True)

        stylized_feats = adaptive_instance_normalization(content_feats, style_feats)
        stylized_feats = alpha * stylized_feats + (1 - alpha) * content_feats

        stylized_image = decoder(stylized_feats)

    return stylized_image

def save_image(image, path):
    image = image.cpu().clone().squeeze(0).clamp(0, 1)
    image = transforms.ToPILImage()(image)
    image.save(path)

@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    result_image = None
    content_filename = None
    style_filename = None
    error = None

    if request.method == 'POST':
        if form.validate_on_submit():
            # Process Content Image
            if form.content.data and form.content.data.filename:
                if allowed_file(form.content.data.filename):
                    content_filename = secure_filename(form.content.data.filename)
                    form.content.data.save(os.path.join(app.config['UPLOAD_FOLDER'], content_filename))
                    form.content_path.data = content_filename
            else:
                content_filename = form.content_path.data

            # Process Style Image
            if form.style.data and form.style.data.filename:
                if allowed_file(form.style.data.filename):
                    style_filename = secure_filename(form.style.data.filename)
                    form.style.data.save(os.path.join(app.config['UPLOAD_FOLDER'], style_filename))
                    form.style_path.data = style_filename
            else:
                style_filename = form.style_path.data

            # Execute Style Transfer
            if content_filename and style_filename:
                content_full_path = os.path.join(app.config['UPLOAD_FOLDER'], content_filename)
                style_full_path = os.path.join(app.config['UPLOAD_FOLDER'], style_filename)

                try:
                    content_image = Image.open(content_full_path).convert('RGB')
                    style_image = Image.open(style_full_path).convert('RGB')

                    alpha_val = float(form.alpha.data) if form.alpha.data is not None else 1.0
                    stylized_tensor = style_transfer(content_image, style_image, encoder, decoder, alpha_val, device)

                    result_filename = 'stylized_' + content_filename
                    result_full_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
                    save_image(stylized_tensor, result_full_path)

                    result_image = result_filename
                except Exception as e:
                    error = f"Style transfer error: {str(e)}"
            else:
                if not content_filename:
                    error = 'Please upload a Content image.'
                elif not style_filename:
                    error = 'Please upload a Style image.'

    return render_template(
        'index.html',
        form=form,
        result_image=result_image,
        content_image=content_filename,
        style_image=style_filename,
        error=error
    )

@app.route('/uploads/<filename>')
def send_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)