from flask import Flask, render_template, request
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models

app = Flask(__name__)

# Load the pre-trained ResNet-18 model
model = models.resnet18(pretrained=True)
num_classes = 2  # Change this to match the number of classes in your model
model.fc = torch.nn.Linear(model.fc.in_features, 1000)

# Load the trained model's parameters
model.load_state_dict(torch.load("E:/COVID-19 DETECTION/Classification using ResNet-18/covid_classification_model2.pth", map_location=torch.device('cpu')))
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    output = model(image)
    probabilities = torch.softmax(output, dim=1)
    _, predicted_class = torch.max(probabilities, 1)
    return predicted_class.item()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', message='No file part')

    file = request.files['file']
    
    if file.filename == '':
        return render_template('index.html', message='No selected file')

    image_path = "uploaded_image.jpg"
    file.save(image_path)

    try:
        predicted_class = predict_image(image_path)
        if predicted_class == 0:
            prediction = "COVID-19 Positive"
        else:
            prediction = "COVID-19 Negative"
        return render_template('index.html', prediction=prediction)
    except Exception as e:
        return render_template('index.html', message='Error predicting image: {}'.format(str(e)))

if __name__ == '__main__':
    app.run(debug=True)
