# import os
import sys
import torch
import logging
from torchvision import transforms
from datetime import datetime
from flask import Flask, request, render_template, jsonify
from model.BCNN import create_bcnn_model
from PIL import Image

classes = [
    "Laysan Albatross",
    "Brewer Blackbird",
    "Painted Bunting",
    "Yellow breasted Chat",
    "European Goldfinch",
    "Herring Gull",
    "Rufous Hummingbird",
    "Blue Jay",
    "Horned Puffin",
    "Pileated Woodpecker"
]


input_size = 448
model_path = "/data1/lianjiawei/pc/exp/cub/exp/init_checkpoints/model_best.pth.tar"

data_transforms = transforms.Compose(
            [
                transforms.Resize(448),
                transforms.CenterCrop(448),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

# load the model
model = create_bcnn_model(
    ["vgg"],
    10,
    "outer_product",
    False,
    True,
    8192,
    2,
    m_sqrt_iter=0,
    proj_dim=0,
)

model = torch.nn.DataParallel(model)
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint["state_dict"])
model.eval()

def preprocess_img(img):
    
    img = img.convert('RGB')
    img = data_transforms(img).unsqueeze(0)

    return img


def predict(model, img):
    with torch.no_grad():
        output = model(img)
    return output.argmax(1).item()


app = Flask(__name__, template_folder='templates')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/up_photo', methods=['post'])
def up_photo():
    img = request.files['file']   
    path = "./static/"
    file_path = path+img.filename
    img.save(file_path)

    image = Image.open(file_path)
    image = preprocess_img(image)
    result = predict(model, image)

    return jsonify({
		"message": "Success!",
        "result": classes[result],
	})
    


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8080)
