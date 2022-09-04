from flask import Flask, request, jsonify
from PIL import Image
from fastai.vision.all import *

app = Flask(__name__)

app.MODEL_PATH = 'export_leaf_classifier.pkl'
app.learner = load_learner(app.MODEL_PATH)

print('DONE LOADING MODEL')

@app.route('/')
def home():
    return 'Home Path for Leaf Disease Detection'

@app.route('/predict', methods=["POST"])
def predict():
    file = request.files['image']

    img = Image.open(file.stream)

    img.save('test_image.jpeg')

    res = app.learner.predict('test_image.jpeg')
    
    category = res[0]
    index = res[1].item()
    probability = res[2].data[index].item()
    #print(category, index, probability)

    return jsonify([category, index, probability])


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
