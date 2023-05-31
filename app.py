from bson import objectid, json_util
from flask import Flask, Response, request, json, jsonify
from flask_cors import CORS, cross_origin
import json

import logging as log

import src, main


# set up logging
log.basicConfig(
    filename='app.log', 
    filemode='a', 
    level=log.DEBUG, 
    format='%(asctime)s %(levelname)s:\n%(message)s\n'
)

app = Flask(__name__)
CORS(app, support_credentials=True)


@app.route('/ping/', methods=['GET'])
def read():
    return Response("pong",
                    status=201,
                    mimetype='application/json')


@app.route('/ask/', methods=['POST'])
@cross_origin()
def write():
    print("Writing Data")
    print(request)
    content_type = request.headers.get('Content-Type')
    print(content_type)
    data = json.loads(request.data.decode('utf-8'))
    print(data)
    result1 = src.run(data['text'])
    result2 = main.run(data['text'])
    try:
        result2 = result2[0]
    except Exception as e:
        print(e)
    return Response(response=json.dumps({
                                            "result1": result1,
                                            "result2": result2
                                        }),
                    status=201,
                    mimetype='application/json')



if __name__ == "__main__":
    app.run(debug=True, use_reloader=True, host='0.0.0.0', port=5002)




