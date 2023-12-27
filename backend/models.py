from uuid import uuid4
from flask_sqlalchemy import SQLAlchemy

# create the extension
db = SQLAlchemy()

def get_uuid():
    return uuid4().hex

class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True, unique = True, autoincrement=True)
    email = db.Column(db.String(150), unique = True)
    password = db.Column(db.String(150), nullable = False)