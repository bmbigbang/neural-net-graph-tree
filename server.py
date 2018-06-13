from flask import render_template, Flask
from model_loader import TFModel
import connexion
import json

app = Flask(__name__)

# Create the application instance
#app = connexion.App(__name__, specification_dir='./')
tf_model = TFModel("./tmp/tfmodel/")
# Read the swagger.yml file to configure the endpoints
## app.add_api('swagger.yml')


@app.route('/')
def home():
    """
    This function just responds to the browser ULR
    localhost:5000/
    :return:        the rendered template 'home.html'
    """

    out = json.dumps(tf_model.layers(), separators=(',', ':'),
               sort_keys=True, indent=4)
    return out


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)