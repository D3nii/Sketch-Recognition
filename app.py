from flask import Flask, Response, jsonify, request

import numpy as np

from main import classifyDrawing
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/checkDrawing', methods=['POST'])
def get_image():
    image = request.values['imageBase64']

    answer = classifyDrawing(image)

    return answer, 200


@app.route('/success', methods=['POST'])
def idek():
    uploaded_file = request.files['file']

    uploaded_file.save("LOL.png")

    print(uploaded_file.__dict__)

    # nparr = np.fromstring(str(uploaded_file), np.uint8)
    # # decode image
    # img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # print(uploaded_file)

    return "uploaded_file",  200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4500, debug=True, threaded=False)
