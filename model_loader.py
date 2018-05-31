import tensorflow as tf
import pickle

weights_dict = {}
train_graph = tf.Graph()
with tf.Session(graph=train_graph) as sess:
    saver = tf.train.import_meta_graph('./tmp/tfmodel/ens4_adv_inception_v3.ckpt.meta')

    saver.restore(sess, "./tmp/tfmodel/ens4_adv_inception_v3.ckpt")

    for n in tf.all_variables():
        if n.name.endswith('weights:0'):
            print(n.name, n.shape)
            split_string = n.name.split("/")
            weights_tuple = split_string[1:split_string.index('weights:0')]

            target = weights_dict
            for i, j in enumerate(weights_tuple):
                if j not in target:
                    target[j] = {}

                if i == len(weights_tuple) - 1:
                    target[j] = {
                        'shape': n.shape,
                        'name': n.name,
                        'array': train_graph.get_tensor_by_name(n.name).eval()
                    }
                else:
                    target = target[j]

pickle.dump(weights_dict, open('weights.pickle', 'wb'))
