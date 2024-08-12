from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/reg-face', methods=['POST'])
def reg_fac():
    data = request.get_json()
    imageBase64 = data['imageBase64']
    #return {"code": "1", "message": "Success"}, 200


if __name__ == '__main__':
    app.run(debug=True)
