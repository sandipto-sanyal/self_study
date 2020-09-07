# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [START gae_python37_app]
from flask import Flask, request, jsonify, make_response


# If `entrypoint` is not defined in app.yaml, App Engine will look for an app
# called `app` in `main.py`.
app = Flask(__name__)


@app.route('/')
def hello():
    """Return a friendly HTTP greeting."""
    message = 'You are in home page!'
    response_json = {'message':message}
    status_code = 200
    return make_response(jsonify(response_json),status_code)

@app.route('/train', methods=['POST'])
def training():
    # read the file from request whose key is 'file'
    file = request.files['file']
    # read the username whose key is 'username'
    username = request.form['username']
    try:
        # train on the uploaded file
        from train import Train
        tr = Train(file=file,username=username)
        mse, r2 = tr.main()
        
        # build the response
        message = 'Hi {}! Training completed'.format(username)
        status_code = 200
        response_json = {'message':message,
                         'mean_sq_error': mse,
                         'r-square': r2
                         }
        return make_response(jsonify(response_json),status_code)
    except Exception as e:
        # build the error response
        message = 'Hi {}! Following error occured while training'.format(username)
        status_code = 500
        response_json = {'message':message,
                         'exception':str(e)
                         }
        return make_response(jsonify(response_json),status_code)
    
@app.route('/predict', methods=['POST'])
def prediction():
    # read the file from request whose key is 'file'
    file = request.files['file']
    # read the username whose key is 'username'
    username = request.form['username']
    try:
        # train on the uploaded file
        from prediction import Prediction
        pr = Prediction(prediction_file=file,username=username)
        prediction_result = pr.main()
        
        # build the response
        message = 'Hi {}! Following are the prediction results completed'.format(username)
        status_code = 200
        response_json = {'message':message,
                         'prediction_result':prediction_result
                         }
        return make_response(jsonify(response_json),status_code)
    except Exception as e:
        # build the error response
        message = 'Hi {}! Following error occured while predicting'.format(username)
        status_code = 500
        response_json = {'message':message,
                         'exception': str(e)
                         }
        return make_response(jsonify(response_json),status_code)
        

if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=False)
# [END gae_python37_app]
