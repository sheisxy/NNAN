import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import model as model
from utils import *

flags = tf.app.flags
flags = tf.app.flags

# Directories
flags.DEFINE_string('data_dir', './data/', 'Directory to store input dataset')

# Run Settings
flags.DEFINE_string('dataset_name', 'mnist', 'The name of dataset as an example')

# Model Settings
flags.DEFINE_integer('input_size', 209, 'The dimension of input')
flags.DEFINE_integer('output_size', 2, 'The dimension of output')
flags.DEFINE_integer('A_node', 1, 'The size of hidden unit in Attention layer')
flags.DEFINE_integer('set_seed', 1, 'The default random seed')
flags.DEFINE_integer('L_node', 128, 'The size of hidden unit in learning module')
flags.DEFINE_float('moving_average_decay', 0.5, 'The average decay rate of moving')

# Training & Optimizer
flags.DEFINE_float('regularization_rate', 0.0001, 'The rate of regularization in Loss Function')
flags.DEFINE_float('learning_rate_base', 0.8, 'The base of learning rate')
flags.DEFINE_float('learning_rate_decay', 0.99, 'The decay of learning rate')
flags.DEFINE_integer('batch_size', 3000, 'The size of batch for minibatch training')
flags.DEFINE_integer('train_step', 100, 'The size of training step')

FLAGS = tf.app.flags.FLAGS

def run_train(sess, train_X, train_Y, val_X, val_Y):
    X = tf.get_collection('input')[0]
    Y = tf.get_collection('output')[0]
    Iterator = BatchCreate(train_X, train_Y)
    for step in range(1, FLAGS.train_step+1):
        if step % 100 == 0:
            val_loss, val_auc, val_aupr = sess.run(tf.get_collection('validate_ops'),
                                                       feed_dict={X:val_X, Y:val_Y})
            print('[%4d] AFS-loss:%.12f AFS-auc:%.6f AFS-auc:%.6f'%
                        (step, val_loss, val_auc, val_aupr))
        xs, ys = Iterator.next_batch(FLAGS.batch_size)

        _, A = sess.run(tf.get_collection('train_ops'), feed_dict={X:xs, Y:ys})
    return A

def run_test(A,train_X, train_Y,test_X, test_Y,total_batch):

    attention_weight = A.mean(0)

    # draw_weight_line_cahrt(attention_weight)
    AFS_wight_rank = list(np.argsort(attention_weight))[::-1]
    auc_score_list = []
    aupr_score_list = []
    index = 1
    # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40]
    for K in range(1,len(AFS_wight_rank), 5):
        use_train_x = train_X[:, AFS_wight_rank[:K]]
        use_test_x = test_X[:, AFS_wight_rank[:K]]
        auc, aupr = model.test(K, use_train_x, train_Y, use_test_x, test_Y, total_batch,index)
        index += 1
        # print('Using Top {} features| auc:{:.4f}'.format(K, auc))

        auc_score_list.append((K, auc))
        aupr_score_list.append((K, aupr))
    return auc_score_list, aupr_score_list
def main(argv=None):

    dataset = 'GPCR'
    data_dir = './data/GPCR'
    intMat, A_sim, B_sim = load_data_from_file(dataset, data_dir)
    # intMat = intMat.T
    cvs = 1
    seeds = [1234]

    if cvs == 1:  # CV setting CVS1
        cv_data = cross_validation1(intMat, seeds, A_sim, B_sim)
    if cvs == 2:  # CV setting CVS2
        cv_data = cross_validation23(intMat, seeds, A_sim)
    if cvs == 3:  # CV setting CVS3
        cv_data = cross_validation23(intMat.T, seeds, B_sim)

    auc_score_k_list, auc_score_list = [], []
    aupr_score_k_list, aupr_score_list = [], []
    for seed in cv_data.keys():
        for train_X, train_Y, test_X, test_Y in cv_data[seed]:
            tf.reset_default_graph()
            Train_size = len(train_X)
            FLAGS.input_size = train_X.shape[1]
            FLAGS.batch_size = int(Train_size / 100)
            print(train_X.shape)
            total_batch = Train_size / FLAGS.batch_size
            model.build(total_batch)

            with tf.Session() as sess:
                init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
                sess.run(init)
                print('== Get feature weight by using AFS ==')
                A = run_train(sess, train_X, train_Y, test_X, test_Y)
            print('==  The Evaluation of AFS ==')
            auc_score_list, aupr_score_list = run_test(A, train_X, train_Y, test_X, test_Y, total_batch)
            auc_score_k_list.append(auc_score_list)
            aupr_score_k_list.append(aupr_score_list)
    auc_score_k_list = np.array(auc_score_k_list)
    aupr_score_k_list = np.array(aupr_score_k_list)
    print(auc_score_k_list.shape)
    for i in range(auc_score_k_list.shape[1]):
        K, auc_vec = auc_score_k_list[:, i][:, 0], auc_score_k_list[:, i][:, 1]
        K, aupr_vec = aupr_score_k_list[:, i][:, 0], aupr_score_k_list[:, i][:, 1]
        # auc_avg, auc_conf = mean_confidence_interval(auc_vec)
        auc_avg = np.mean(auc_vec)
        aupr_avg = np.mean(aupr_vec)
        print("Using Top %4d features| auc:%.6f auc:%.6f" % (K[0], auc_avg, aupr_avg))

if __name__ == '__main__':
    tf.app.run()