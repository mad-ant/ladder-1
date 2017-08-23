import tensorflow as tf
import math
import os
import csv
from tqdm import tqdm
import input_generic_data
import numpy as np
import sklearn.metrics
import pandas as pd

layer_sizes = [55, 50, 45, 30, 20, 10, 2]

L = len(layer_sizes) - 1  # number of layers

num_examples = 30000
num_epochs = 5
num_labeled = 10000

starter_learning_rate = 0.02
decay_after = 15  # epoch after which to begin learning rate decay

batch_size = 100
num_iter = (num_examples/batch_size) * num_epochs  # number of loop iterations

inputs = tf.placeholder(tf.float32, shape=(None, layer_sizes[0]),name = 'x')
outputs = tf.placeholder(tf.float32,name = 'labels')

def input_fn(data_file, num_epochs, shuffle):
  """Input builder function."""
  tmp_data = pd.read_csv(
      tf.gfile.Open(data_file),
      header=None)
  tmp_data = tmp_data.dropna(how="any", axis=0)
  labels = tmp_data.iloc[:,0]
  features = match_data.iloc[:,1:]
  return tf.estimator.inputs.pandas_input_fn(
      x=features,
      y=labels,
      batch_size=100,
      num_epochs=num_epochs,
      shuffle=shuffle,
      num_threads=5)


def bi(inits, size, name):
    with tf.name_scope ("bi",name) as scope:
        return tf.Variable(inits * tf.ones([size]), name=scope)


def wi(shape, name):
    with tf.name_scope("wi",name) as scope:
        return tf.Variable(tf.random_normal(shape, name=scope)) / math.sqrt(shape[0])

shapes = zip(layer_sizes[:-1], layer_sizes[1:])  # shapes of linear layers

with tf.name_scope("weights"):
    weights = {'W': [wi(s, "W") for s in shapes],  # Encoder weights
               'V': [wi(s[::-1], "V") for s in shapes],  # Decoder weights
               # batch normalization parameter to shift the normalized value
               'beta': [bi(0.0, layer_sizes[l+1], "beta") for l in range(L)],
               # batch normalization parameter to scale the normalized value
               'gamma': [bi(1.0, layer_sizes[l+1], "beta") for l in range(L)]}

noise_std = 0.1  # scaling factor for noise used in corrupted encoder

# hyperparameters that denote the importance of each layer
denoising_cost = [1, 1, 0.1, 0.1, 0.01,0.01,0.01]

join = lambda l, u: tf.concat([l, u], 0)
labeled = lambda x: tf.slice(x, [0, 0], [batch_size, -1]) if x is not None else x
unlabeled = lambda x: tf.slice(x, [batch_size, 0], [-1, -1]) if x is not None else x
split_lu = lambda x: (labeled(x), unlabeled(x))

training = tf.placeholder(tf.bool)

ewma = tf.train.ExponentialMovingAverage(decay=0.99,name="exponential_moving_average")  # to calculate the moving averages of mean and variance
bn_assigns = []  # this list stores the updates to be made to average mean and variance


def batch_normalization(batch, mean=None, var=None,name=None):
    with tf.name_scope(name,"corrupted_batch_normalization") as scope:
        if mean is None or var is None:
            mean, var = tf.nn.moments(batch, axes=[0],name=scope)
        return (batch - mean) / tf.sqrt(var + tf.constant(1e-10))

# average mean and variance of all layers
with tf.name_scope("running_mean"):
    running_mean = [tf.Variable(tf.constant(0.0, shape=[l]), trainable=False,name='rm_layer_'+str(idx)) for (idx,l) in enumerate(layer_sizes[1:])]
with tf.name_scope("running_var"):
    running_var = [tf.Variable(tf.constant(1.0, shape=[l]), trainable=False,name='rm_var_'+str(idx)) for (idx,l) in enumerate(layer_sizes[1:])]


def update_batch_normalization(batch, l):
    "batch normalize + update average mean and variance of layer l"
    with tf.name_scope("layer_"+str(l)):
        mean, var = tf.nn.moments(batch, axes=[0],name='clean_batch_normalization')
        assign_mean = running_mean[l-1].assign(mean)
        assign_var = running_var[l-1].assign(var)
        bn_assigns.append(ewma.apply([running_mean[l-1], running_var[l-1]]))
        with tf.control_dependencies([assign_mean, assign_var]):
            return (batch - mean) / tf.sqrt(var + 1e-10)


def encoder(inputs, noise_std):
    with tf.name_scope("h_0"):
        h = inputs + tf.random_normal(tf.shape(inputs)) * noise_std  # add noise to input
    d = {}  # to store the pre-activation, activation, mean and variance for each layer
    # The data for labeled and unlabeled examples are stored separately
    d['labeled'] = {'z': {}, 'm': {}, 'v': {}, 'h': {}}
    d['unlabeled'] = {'z': {}, 'm': {}, 'v': {}, 'h': {}}
    d['labeled']['z'][0], d['unlabeled']['z'][0] = split_lu(h)
    encodertype =  "corrupted_enc_" if (noise_std != 0.0) else "clean_enc_"
    for l in range(1, L+1):
        with tf.name_scope(encodertype+str(l)):
            print "Layer ", l, ": ", layer_sizes[l-1], " -> ", layer_sizes[l]
            d['labeled']['h'][l-1], d['unlabeled']['h'][l-1] = split_lu(h)
            z_pre = tf.matmul(h, weights['W'][l-1])  # pre-activation
            z_pre_l, z_pre_u = split_lu(z_pre)  # split labeled and unlabeled examples

            m, v = tf.nn.moments(z_pre_u, axes=[0],name='encoder_batch_normalization')

            # if training:
            def training_batch_norm():
                with tf.name_scope(encodertype+str(l)):
                    # Training batch normalization
                    # batch normalization for labeled and unlabeled examples is performed separately
                    if noise_std > 0:
                        # Corrupted encoder
                        # batch normalization + noise
                        z = join(batch_normalization(z_pre_l, name=encodertype+str(l)), batch_normalization(z_pre_u, m, v, name=encodertype+str(l)))
                        z += tf.random_normal(tf.shape(z_pre),name="gaussian_noise") * noise_std
                    else:
                        # Clean encoder
                        # batch normalization + update the average mean and variance using batch mean and variance of labeled examples
                        z = join(update_batch_normalization(z_pre_l, l), batch_normalization(z_pre_u, m, v))
                    return z

            # else:
            def eval_batch_norm():
                with tf.name_scope(encodertype+str(l)):
                    # Evaluation batch normalization
                    # obtain average mean and variance and use it to normalize the batch
                    mean = ewma.average(running_mean[l-1])
                    var = ewma.average(running_var[l-1])
                    z = batch_normalization(z_pre, mean, var)
                    # Instead of the above statement, the use of the following 2 statements containing a typo
                    # consistently produces a 0.2% higher accuracy for unclear reasons.
                    # m_l, v_l = tf.nn.moments(z_pre_l, axes=[0])
                    # z = join(batch_normalization(z_pre_l, m_l, mean, var), batch_normalization(z_pre_u, mean, var))
                    return z

            # perform batch normalization according to value of boolean "training" placeholder:
            with tf.name_scope("trainOrtest"):
                z = tf.cond(training, training_batch_norm, eval_batch_norm)

            if l == L:
                # use softmax activation in output layer
                with tf.name_scope("output_softmax"):
                    h = tf.nn.softmax(weights['gamma'][l-1] * (z + weights["beta"][l-1]))
            else:
                with tf.name_scope ("ReLU"):
                    # use ReLU activation in hidden layers
                    h = tf.nn.relu(z + weights["beta"][l-1])

            d['labeled']['z'][l], d['unlabeled']['z'][l] = split_lu(z)
            d['unlabeled']['m'][l], d['unlabeled']['v'][l] = m, v  # save mean and variance of unlabeled examples for decoding
    d['labeled']['h'][l], d['unlabeled']['h'][l] = split_lu(h)
    return h, d

print "=== Corrupted Encoder ==="
y_c, corr = encoder(inputs, noise_std)

print "=== Clean Encoder ==="
y, clean = encoder(inputs, 0.0)  # 0.0 -> do not add noise

print "=== Decoder ==="


def g_gauss(z_c, u, size):
    "gaussian denoising function proposed in the original paper"
    with tf.name_scope("gauss_denoising"):
        wi = lambda inits, name: tf.Variable(inits * tf.ones([size]), name=name)
        a1 = wi(0., 'a1')
        a2 = wi(1., 'a2')
        a3 = wi(0., 'a3')
        a4 = wi(0., 'a4')
        a5 = wi(0., 'a5')

        a6 = wi(0., 'a6')
        a7 = wi(1., 'a7')
        a8 = wi(0., 'a8')
        a9 = wi(0., 'a9')
        a10 = wi(0., 'a10')

        mu = a1 * tf.sigmoid(a2 * u + a3) + a4 * u + a5
        v = a6 * tf.sigmoid(a7 * u + a8) + a9 * u + a10

        z_est = (z_c - mu) * v + mu
        return z_est

# Decoder
z_est = {}
d_cost = []  # to store the denoising cost of all layers
for l in range(L, -1, -1):
    print "Layer ", l, ": ", layer_sizes[l+1] if l+1 < len(layer_sizes) else None, " -> ", layer_sizes[l], ", denoising cost: ", denoising_cost[l]
    z, z_c = clean['unlabeled']['z'][l], corr['unlabeled']['z'][l]
    m, v = clean['unlabeled']['m'].get(l, 0), clean['unlabeled']['v'].get(l, 1-1e-10)
    if l == L:
        u = unlabeled(y_c)
    else:
        u = tf.matmul(z_est[l+1], weights['V'][l])
    u = batch_normalization(u,name="decode_"+str(l))
    z_est[l] = g_gauss(z_c, u, layer_sizes[l])
    z_est_bn = (z_est[l] - m) / v
    # append the cost of this layer to d_cost
    d_cost.append((tf.reduce_mean(tf.reduce_sum(tf.square(z_est_bn - z), 1)) / layer_sizes[l]) * denoising_cost[l])

# calculate total unsupervised cost by adding the denoising cost of all layers
with tf.name_scope("costs"):
    u_cost = tf.add_n(d_cost,name='unsupervised_cost')
    y_N = labeled(y_c)
    cost = -tf.reduce_mean(tf.reduce_sum(outputs*tf.log(y_N), 1),name='supervised_cost')  # supervised cost
    loss = cost + u_cost  # total cost
    loss_summary = tf.summary.scalar("loss",loss)
    u_summary = tf.summary.scalar("unsupervised_cost",u_cost)
    cost_summary = tf.summary.scalar("supervised_cost",cost)
    pred_cost = -tf.reduce_mean(tf.reduce_sum(outputs*tf.log(y), 1))  # cost used for prediction

with tf.name_scope("metrics"):
    predictions = tf.argmax(y, 1)
    actuals = tf.argmax(outputs, 1)
    correct_prediction = tf.equal(predictions,actuals )  # no of correct predictions
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) # * tf.constant(100.0)
    precision,op_prec = tf.contrib.metrics.streaming_precision(predictions,actuals)
    recall,op_recall = tf.contrib.metrics.streaming_recall(predictions,actuals)
#precision = tf.cast(precision,"float")
#recall = tf.cast(precision,"float")


#with tf.name_scope("metrics"):
#    train_a_summary = tf.summary.scalar("train_accuracy",accuracy)
    #train_p_summary = tf.summary.scalar("train_precision",precision)
    #train_r_summary = tf.summary.scalar("train_recall", recall)

learning_rate = tf.Variable(starter_learning_rate, trainable=False,name='learning_rate')
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)


# add the updates of batch normalization statistics to train_step
bn_updates = tf.group(*bn_assigns)
with tf.control_dependencies([train_step]):
    train_step = tf.group(bn_updates)

print "===  Loading Data ==="

#traindata = input_fn("/Users/tdawn/Documents/wrk/data/m5.ladder.train.csv",10,True)
#testdata = input_fn("/Users/tdawn/Documents/wrk/data/m5.ladder.test.csv",10,True)
matchdb = input_generic_data.read_data_sets("SBO_data", n_labeled=num_labeled, one_hot=True)

saver = tf.train.Saver()

print "===  Starting Session ==="
sess = tf.Session()

i_iter = 0

ckpt = tf.train.get_checkpoint_state('checkpoints_matching/')  # get latest checkpoint (if any)
if ckpt and ckpt.model_checkpoint_path:
    # if checkpoint exists, restore the parameters and set epoch_n and i_iter
    saver.restore(sess, ckpt.model_checkpoint_path)
    epoch_n = int(ckpt.model_checkpoint_path.split('-')[1])
    i_iter = (epoch_n+1) * (num_examples/batch_size)
    print "Restored Epoch ", epoch_n
else:
    # no checkpoint exists. create checkpoints directory if it does not exist.
    if not os.path.exists('checkpoints_matching'):
        os.makedirs('checkpoints_matching')
    init = tf.global_variables_initializer()
    initl = tf.local_variables_initializer()
    sess.run(init)
    sess.run(initl)

summ = tf.summary.merge_all()
writer = tf.summary.FileWriter("/Users/tdawn/Documents/wrk/tbdir/ladder-wrk/ladder.test.tb.4")
#val_writer = tf.summary.FileWriter("/Users/tdawn/Documents/wrk/tbdir/ladder/ladder.test.tb.scaled.noise-metrics-test")
writer.add_graph(sess.graph)

#test_accuracy = tf.summary.scalar("test_accuracy",accuracy)
#test_precision = tf.summary.scalar("test_precision",precision)
#test_recall = tf.summary.scalar("test_recall",recall)
#tf.summary.merge(test_accuracy,[test_summ])
#tf.summary.merge(test_precision,[test_summ])
#tf.summary.merge(test_recall,[test_summ])

print "=== Training ==="
acc_,prec_,recall_ = sess.run([accuracy,op_prec,op_recall], feed_dict={inputs: matchdb.test.features, outputs: matchdb.test.labels, training: False})
init_prec, init_recall = sess.run([precision,recall])
print "Initial Numbers: ", acc_, init_prec,init_recall
#y_gte = tf.greater_equal(y)
y_p = tf.argmax(y,1)
y_true = np.argmax(matchdb.test.labels,1)
val_accuracy, y_pred = sess.run([accuracy,y_p], feed_dict={inputs: matchdb.test.features, outputs: matchdb.test.labels, training: False})
print "Initial:\nFScore: %2.3f\nPrecision: %2.3f\nRecall: %2.3f\nAccuracy: %2.3f\n" % (100.0*sklearn.metrics.f1_score(y_true,y_pred) ,
                    100.0*sklearn.metrics.precision_score(y_true,y_pred),
                    100.0*sklearn.metrics.recall_score(y_true,y_pred),
                    100.0*sklearn.metrics.accuracy_score(y_true,y_pred)) 
print sklearn.metrics.confusion_matrix(y_true,y_pred)


for i in tqdm(range(i_iter, num_iter)):
    features, labels = matchdb.train.next_batch(batch_size)
    _,sss = sess.run([train_step,summ], feed_dict={inputs: features, outputs: labels, training: True})
    #sss = sess.run([summ], feed_dict={inputs: features, outputs: labels, training: False})
    writer.add_summary(sss, i)
    if (i > 1) and ((i+1) % (num_iter/num_epochs) == 0):
        epoch_n = i/(num_examples/batch_size)
        if (epoch_n+1) >= decay_after:
            # decay learning rate
            # learning_rate = starter_learning_rate * ((num_epochs - epoch_n) / (num_epochs - decay_after))
            ratio = 1.0 * (num_epochs - (epoch_n+1))  # epoch_n + 1 because learning rate is set for next epoch
            ratio = max(0, ratio / (num_epochs - decay_after))
            sess.run(learning_rate.assign(starter_learning_rate * ratio))
        saver.save(sess, 'checkpoints/model.ckpt', epoch_n)
        # print "Epoch ", epoch_n, ", Accuracy: ", sess.run(accuracy, feed_dict={inputs: matchdb.test.features, outputs:matchdb.test.labels, training: False}), "%"
        with open('train_log', 'ab') as train_log:
            # write test accuracy to file "train_log"
            train_log_w = csv.writer(train_log)
            acc_n = sess.run([accuracy], feed_dict={inputs: matchdb.test.features, outputs: matchdb.test.labels, training: False})
            log_i = [epoch_n] + acc_n
            train_log_w.writerow(log_i)


#Final Metrics

acc_,prec_,recall_ = sess.run([accuracy,op_prec,op_recall], feed_dict={inputs: matchdb.test.features, outputs: matchdb.test.labels, training: False})
final_prec, final_recall = sess.run([precision,recall])
print "Final Numbers: ", acc_, final_prec,final_recall

y_p = tf.argmax(y,1)
val_accuracy, y_pred = sess.run([accuracy,y_p], feed_dict={inputs: matchdb.test.features, outputs: matchdb.test.labels, training: False})
y_true = np.argmax(matchdb.test.labels,1) 
print "Final:\nFScore: %2.3f\nPrecision: %2.3f\nRecall: %2.3f\nAccuracy: %2.3f\n" % (100.0*sklearn.metrics.f1_score(y_true,y_pred) ,
                    100.0*sklearn.metrics.precision_score(y_true,y_pred),
                    100.0*sklearn.metrics.recall_score(y_true,y_pred),
                    100.0*sklearn.metrics.accuracy_score(y_true,y_pred))
print sklearn.metrics.confusion_matrix(y_true,y_pred)
#print "Final Accuracy: ", sess.run([accuracy,op_prec,op_recall], feed_dict={inputs: matchdb.test.features, outputs: matchdb.test.labels, training: False}), "%"

sess.close()
