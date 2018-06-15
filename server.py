from model_loader import TFModel
import connexion

app = connexion.App(__name__, specification_dir='./')

tf_model = TFModel("./tmp/tfmodel/")


# @app.route('/')
def layers():
    """
    This function just responds to the browser ULR
    localhost:5000/
    :return:        the rendered template 'home.html'
    """

    return {'layers': tf_model.layers()}


app.add_api('swagger.yaml')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)