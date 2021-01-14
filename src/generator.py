import tensorflow as tf
tf.compat.v1.disable_eager_execution()

class Generator(object):
    def __init__(self, n_node, node_emb_init, positive, config):
        self.n_node = n_node
        self.node_emb_init = node_emb_init

        with tf.compat.v1.variable_scope('generator'):
            self.embedding_matrix = tf.compat.v1.get_variable(name="embedding_generator", shape=self.node_emb_init.shape, trainable=True)
            self.bias_vector = tf.Variable(tf.zeros([self.n_node]))

        self.node_id = tf.compat.v1.placeholder(tf.int32, shape=[None])
        self.node_neighbor_id = tf.compat.v1.placeholder(tf.int32, shape=[None])
        self.reward = tf.compat.v1.placeholder(tf.float32, shape=[None])
        self.node_embedding = tf.nn.embedding_lookup(params=self.embedding_matrix, ids=self.node_id)  # batch_size * n_embed

        self.target_node = tf.compat.v1.placeholder(tf.int32, shape=[None])
        self.target_embedding = tf.nn.embedding_lookup(params=self.embedding_matrix, ids=self.target_node)
        self.target_score = tf.matmul(self.target_embedding, self.embedding_matrix, transpose_b=True) + self.bias_vector

        self.node_neighbor_embedding = tf.nn.embedding_lookup(params=self.embedding_matrix, ids=self.node_neighbor_id)
        self.bias = tf.gather(self.bias_vector, self.node_neighbor_id)

        self.score = tf.reduce_sum(input_tensor=self.node_embedding * self.node_neighbor_embedding, axis=1) + self.bias
        self.prob = tf.clip_by_value(tf.nn.sigmoid(self.score), 1e-5, 1)

        self.pairs_relevances = tf.compat.v1.placeholder(tf.float32, shape=[None]) 
        self.pairs_score = tf.clip_by_value(tf.nn.sigmoid(self.pairs_relevances), 1e-5, 1)
        if positive == True:
            self.loss = -tf.reduce_mean(input_tensor=tf.math.log(self.prob) * self.reward) + config.lambda_gen * (
                    tf.nn.l2_loss(self.node_neighbor_embedding) + tf.nn.l2_loss(self.node_embedding))
        else:
            self.loss = tf.reduce_mean(input_tensor=tf.math.log(self.prob) * self.reward) + config.lambda_gen * (
                    tf.nn.l2_loss(self.node_neighbor_embedding) + tf.nn.l2_loss(self.node_embedding))

        optimizer = tf.compat.v1.train.GradientDescentOptimizer(config.lr_gen)
        self.g_updates = optimizer.minimize(self.loss)
