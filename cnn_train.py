import os
import sys
import time
from datetime import timedelta

import numpy as np
import tensorflow as tf

from cnn_model import TCNNConfig, TextCNN
from input_data import load_data_label

train_list_side = None
train_list_tag = None
val_list_side = None
val_list_tag = None
test_list_side = None
test_list_tag = None

Top_k = 20

save_dir = 'model/'
save_path = os.path.join(save_dir, 'cnn_model')

def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def feed_data(x_batch, y_batch):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
    }
    return feed_dict

def batch_iter(x, y, batch_size=32):
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]

def evaluate(sess, x_, y_):
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 128)
    total_loss = 0.0
    total_map = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch)
        loss, y_pred = sess.run([model.loss, model.y_pred], feed_dict=feed_dict)
        map = MAP(y_pred, y_batch)
        total_loss += loss * batch_len
        total_map += map * batch_len

    return total_loss / data_len, total_map / data_len

def MAP(y_p, y_t):
    data_len = len(y_p)
    map = 0
    for yp, yt in zip(y_p, y_t):
        ids = np.argsort(-yp)[:Top_k]
        count = 1
        true_num = 1
        map_each = 0
        for id in ids:
            if yt[count]>0:
                map_each += true_num/count
                true_num += 1
            count+=1
        map += map_each/Top_k
    return map/data_len

def train():
    print("Configuring TensorBoard and Saver...")
    tensorboard_dir = 'tensorboard/cnn'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    tf.summary.scalar("loss", model.loss)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0
    best_map_val = 0.0
    last_improved = 0
    require_improvement = 2000

    flag = False
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter(train_list_side, train_list_tag, config.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(x_batch, y_batch)

            if total_batch % config.save_per_batch == 0:
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % config.print_per_batch == 0:

                loss_train, y_pred = session.run([model.loss, model.y_pred], feed_dict=feed_dict)
                loss_val, map_val = evaluate(session, val_list_side, val_list_tag)  # todo

                map_train = MAP(y_pred, y_batch)

                if map_val > best_map_val:
                    best_map_val = map_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train MAP: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val MAP: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, map_train, loss_val, map_val, time_dif, improved_str))

            session.run(model.optim, feed_dict=feed_dict)
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                print("No optimization for a long time, early-stopping...")
                flag = True
                break
        if flag:
            break

def test():
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)

    print('Testing...')
    loss_test, map_test = evaluate(session, test_list_side, test_list_tag)
    msg = 'Test Loss: {0:>6.2}, Test MAP: {1:>7.2%}'
    print(msg.format(loss_test, map_test))


if __name__ == '__main__':
    print('Configuring CNN model...')
    config = TCNNConfig()
    train_list_side, train_list_tag, val_list_side, val_list_tag, test_list_side, test_list_tag = load_data_label('')
    model = TextCNN(config)

    if sys.argv[1] == 'train':
        train()
    else:
        test()