
from flask import Flask, request

# Khởi tạo flask app
app = Flask(__name__)
# route và method
@app.route("/", methods=["GET"])
# Hàm xử lý dữ liệu
def _hello_world():
	return "Hello world"


