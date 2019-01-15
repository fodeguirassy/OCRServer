# -*- coding: utf-8 -*-
from flask import Flask, request
from PIL import Image
import numpy as np
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def add_example():
    if request.method == 'POST':
        json = request.get_json(force=True)
        return json['label']
    else:
        return 'It is a GET'


if __name__ == '__main__':
    app.run()
