from flask import Flask, render_template
import forecast_plot  # Make sure this runs and generates the image

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
