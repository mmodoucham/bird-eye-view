from httplib2 import Response
import pandas as pd
from flask import Flask, Response, make_response, jsonify, redirect
from flask_cors import CORS
from src.models.train_model import predict
import base64
import swifter

app = Flask(__name__)
CORS(app)

DATASET_PATH = 'data/images/'

df = pd.read_csv('data/df.csv')


def base64encryption(img):
    with open(DATASET_PATH + img, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        return encoded_string



@app.route('/')
def index():
    start_id = df.sample(n=1)['id'].iloc[0]
    #return Response(df.to_json(orient="records"), mimetype='application/json')
    return redirect('/product/' + str(start_id))



@app.route('/product/<product_id>', methods=['GET'])
def get_product(product_id):
    try:
        product = df[df['id'] == product_id].index.values
        prediction = predict(product)
        print(prediction)
        pred = df.iloc[prediction]
        pred['base64'] = pred['image'].apply(base64encryption)
        return Response(pred.to_json(orient="records"), mimetype='application/json')
    except:
        message = "Product not found"
        status_code = 404
        return make_response(jsonify(message), status_code)


if __name__ == "__main__":
    app.run(debug=True, port=1234)
