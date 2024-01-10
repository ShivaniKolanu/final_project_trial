from uuid import uuid4
from flask_sqlalchemy import SQLAlchemy
import face_recognition
from sqlalchemy.dialects.postgresql import BYTEA

# create the extension
db = SQLAlchemy()

def get_uuid():
    return uuid4().hex

class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True, unique = True, autoincrement=True)
    email = db.Column(db.String(150), unique = True)
    password = db.Column(db.String(150), nullable = False)

class Face(db.Model):
    __tablename__ = "face"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(50), nullable=False)
    face_image = db.Column(BYTEA, nullable=False)

class Faces(db.Model):
    __tablename__ = "faces"
    id = db.Column(db.Integer, primary_key=True, unique = True, autoincrement=True)
    name = db.Column(db.String(255), nullable = False)
    image_path = db.Column(db.String(255), nullable =False)



class tbl_faces(db.Model):
    __tablename__ = "tbl_faces"
    id = db.Column(db.Integer, primary_key = True, unique = True, autoincrement = True)
    name = db.Column(db.String(255), nullable = False)