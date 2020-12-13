import tensorflow as tf

class TCNNConfig(object):
    num_classes = 7067 #aspect
    num_filters = 12
    learning_rate = 1e-3
    batch_size = 64
    num_epochs = 30000
    print_per_batch = 20
    save_per_batch = 1000
    W = 100

class TextCNN(object):
    def __init__(self, config):
        self.config = config

        self.input_x = tf.placeholder(tf.float32, [None, 5, self.config.num_classes, 1], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')

        self.cnn()

    def cnn(self):
        with tf.name_scope("cnn"):
            tf.shape(self.input_x)
            conv1_1 = tf.layers.conv2d(self.input_x, self.config.num_filters, (5, 1), name='conv1_1')
            conv1_2 = tf.layers.conv2d(self.input_x, self.config.num_filters, (4, 1), name='conv1_2')
            conv1_3 = tf.layers.conv2d(self.input_x, self.config.num_filters, (3, 1), name='conv1_3')
            conv1_4 = tf.layers.conv2d(self.input_x, self.config.num_filters, (2, 1), name='conv1_4')

            maxpool1_1 = tf.nn.max_pool(conv1_1, [1, 1, 1, 1], [1, 1, 1, 1], "VALID", name='maxpool1_1')
            maxpool1_2 = tf.nn.max_pool(conv1_2, [1, 2, 1, 1], [1, 1, 1, 1], "VALID", name='maxpool1_2')
            maxpool1_3 = tf.nn.max_pool(conv1_3, [1, 3, 1, 1], [1, 1, 1, 1], "VALID", name='maxpool1_3')
            maxpool1_4 = tf.nn.max_pool(conv1_4, [1, 4, 1, 1], [1, 1, 1, 1], "VALID", name='maxpool1_4')

            concat = tf.concat([maxpool1_1, maxpool1_2, maxpool1_3, maxpool1_4], 1)
            fc1 = tf.layers.dense(concat, 1, name='fc1')
            conv2 = tf.layers.conv2d(fc1, self.config.num_filters, (2, 1), name='conv2')
            maxpool2 = tf.nn.max_pool(conv2, [1, 3, 1, 1], [1, 1, 1, 1], "VALID", name='maxpool1_4')

        with tf.name_scope("score"):
            fc = tf.layers.dense(maxpool2, 1, name='fc2')
            self.y_pred = tf.reshape(fc, (-1, self.config.num_classes))

        with tf.name_scope("optimize"):
            cross_entropy = tf.nn.weighted_cross_entropy_with_logits(logits=self.y_pred, pos_weight=self.config.W, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)