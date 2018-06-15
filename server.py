import connexion

from model_loader import TFModel

app = connexion.App(__name__, specification_dir='./')

tf_model = TFModel("./tmp/tfmodel/")


def layers():
    """
    responsible for /layers api path for the selected model
    :return:        layers nested dict of node names
    """
    return {'layers': tf_model.layers()}


def node(name):
    """
    grab the node data for given node name through /node api
    path for the selected model
    :param name:   train graph of node to search in model
    :returns:   array of arrays and shape and name of node
    """
    try:
        return tf_model.node(name)
    except KeyError:
        return {'error': 'given node not found'}, 400


app.add_api('swagger.yaml')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)