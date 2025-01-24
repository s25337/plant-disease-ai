from flask import Flask, render_template, request, redirect, url_for
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
import nbimporter
from Rainforest import RainforestNN


app = Flask(__name__)


UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


MODEL_PATH = 'content/rain_forest.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = RainforestNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No file part", 400

    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    if file:
        # Zapisz przesłane zdjęcie
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        uploaded_image_url = url_for('static', filename=f'uploads/{file.filename}')

        # Otwórz obraz i przetwórz
        img = Image.open(filepath).convert('RGB')
        img = transform(img).unsqueeze(0).to(device)

        # Przewidywanie
        with torch.no_grad():
            output = model(img)
            _, predicted = torch.max(output, 1)
            prediction = 'Zdrowa (1)' if predicted.item() == 1 else 'Chora (0)'

        return render_template('index.html', result=prediction, uploaded_image_url=uploaded_image_url)
       # return f"<h1>Wynik: {prediction}</h1><br><img src='/{filepath}' width='300'>"

if __name__ == '__main__':
    app.run(debug=True)
