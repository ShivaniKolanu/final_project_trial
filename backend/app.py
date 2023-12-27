from flask import Flask, jsonify, request, render_template
from models import db, User
from flask_cors import CORS, cross_origin

app = Flask(__name__)

app.config['SECRET_KEY'] = 'cairocoders-ednalan'
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:KShivani@localhost:5432/react_login'

SQLALCHEMY_TRACK_MODIFICATIONS = False
SQLALCHEMY_ECHO = True

CORS(app, supports_credentials= True)
db.init_app(app)
  
with app.app_context():
    db.create_all()

@app.route("/")
def hello_world():
    return "Hello, World!"

@app.route("/signup", methods = ["POST"])
def signup():

    email = request.json["email"]
    password = request.json["password"]

    data = User.query.filter_by(email = email).first()
    if data:
        return {"error": "User already exist"}, 500
    else:
        new_data = User(email = email, password = password)
        db.session.add(new_data)
        db.session.commit()
        return jsonify({
            "id": new_data.id,
            "email": new_data.email
        })
    
@app.route("/login", methods = ["POST"])
def login():

    email = request.json["email"]
    password = request.json["password"]

    data = User.query.filter_by(email = email, password = password).first()
    if data:
        return jsonify({
            "id": data.id,
            "email": data.email
        }), 200
    else:
        return {"error": "User does not exist or wrong password"}, 500



if __name__ == "__main__":
    app.run(debug = True)