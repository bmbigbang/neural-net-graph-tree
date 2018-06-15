import tensorflow as tf


class TFModel:
    # "./tmp/tfmodel/"
    def __init__(self, relative_path, meta_name="ens4_adv_inception_v3.ckpt.meta",
                 checkpoint="ens4_adv_inception_v3.ckpt"):
        self.relative_path = relative_path
        self.meta_name = meta_name
        self.checkpoint = checkpoint
        self.train_graph = tf.Graph()
        with tf.Session(graph=self.train_graph) as sess:
            self.saver = tf.train.import_meta_graph(relative_path + self.meta_name)

    def layers(self, node_type='weights:0'):
        nodes = {}
        with tf.Session(graph=self.train_graph) as sess:
            for n in tf.all_variables():
                if n.name.endswith(node_type):
                    split_string = n.name.split("/")
                    weights_tuple = split_string[:split_string.index(node_type)]

                    target = nodes
                    for j in weights_tuple:
                        if j not in target:
                            target[j] = {}
                        target = target[j]

        return nodes

    def node(self, name, node_type='weights:0'):
        ret = {'name': name, 'shape': [], 'array': []}
        with tf.Session(graph=self.train_graph) as sess:
            self.saver.restore(sess, self.relative_path + self.checkpoint)
            t = self.train_graph.get_tensor_by_name(name + node_type).eval()
            ret['shape'] = t.tolist()
            ret['array'] = t.shape

        return ret


# json.dump({}, codecs.open(file_path, 'w', encoding='utf-8'),
#           separators=(',', ':'), sort_keys=True, indent=4)

# from model_loader import TFModel
# a = TFModel("./tmp/tfmodel/")
# a.node("InceptionV3/Conv2d_1a_3x3/")