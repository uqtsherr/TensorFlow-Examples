from __future__ import division, print_function, absolute_import


import tensorflow as tf
import h5py
import numpy as np
#import matplotlib.pyplot as plt
import os.path
from os import listdir
from os.path import isfile, join
import time
import edflib

#foldername = 'C:\\Users\\Tim\\EEG Free NAthan'
foldername = 'C:\\Users\\Tim\\eegNathanTest'
filelist = [f for f in listdir(foldername) if isfile(join(foldername, f))]
fs =256
signal_labels = []
signal_nsamples = []
def fileinfo(edf):
    #print("datarecords_in_file", edf.datarecords_in_file)
    #print("signals_in_file:", edf.signals_in_file)

    for ii in range(edf.signals_in_file):
        signal_labels.append(edf.signal_label(ii))
        print("signal_label(%d)" % ii, edf.signal_label(ii), end='')
        print(edf.samples_in_file(ii), edf.samples_in_datarecord(ii), end='')
        signal_nsamples.append(edf.samples_in_file(ii))
        print(edf.samplefrequency(ii))
    return edf.samples_in_file(ii), edf.signals_in_file


def readsignals(edf, nPoints, sig1, buf=None):
    """times in seconds"""
    nsigs = edf.signals_in_file
    for ii in range(nsigs):
        edf.read_digital_signal(ii, 0, nPoints, buf)
        sig1[:, ii] = buf

#takes the label list, produces examples from it
def getLeaveOneOut(LabelsLst, DataLst,N):
    Nrecords = len(DataLst)
    records = []
    for k in range(Nrecords-1):
        name = DataLst[k][1]
        data = DataLst[k][0]

        annotationNum = int(name.strip( join(foldername,'eeg' )))
        print(annotationNum)
        currLabels = LabelsLst[annotationNum]
        clen = np.shape(currLabels)[0]
        n_chann = np.shape(data)[1]
        print(clen)
        XDat = np.zeros((clen,fs,n_chann))
        for cnt in range(clen-1):
            timestep = data[cnt*fs:(cnt+1)*fs][:]
            XDat[cnt,:,:] = timestep


        if k == N:
            validation = XDat
        else:
            records.append((XDat, currLabels))


    return records, validation


print(foldername)
print(filelist)

#note to Elliot, python indexing starts at zero not 1
num_classes = 4
annotations = []
i = 0
if isfile(join(foldername, 'annotations_EEGdata.mat')):
    with h5py.File(join(foldername, 'annotations_EEGdata.mat')) as file:
        for c in file['annotat_new']:
            for r in range(len(c)):
                annotations.append(file[c[r]][()])
                i = i + 1
print(annotations)

eegdata = []
for file in filelist:
    fpath = join(foldername,file)
    p, ext = os.path.splitext(fpath)
    if(ext == '.edf'):
        print(fpath + " is a .edf file")
        edf = edflib.EdfReader(fpath)
        samples, nSigs = fileinfo(edf)
        sig1 = np.zeros((samples,nSigs), dtype='int32')
        buf =  np.zeros(samples, dtype='int32')
        readsignals(edf,samples,sig1,buf)
        datName = (sig1,p)
        eegdata.append(datName)
        print(sig1)
    else:
        print(fpath + " is not a .edf file")


records, validation = getLeaveOneOut(annotations, eegdata, 0);



##End data load section - records is a tuple of labels,Dataset
#Data is currently imported as a record, where records are made up of a (label data) tuple. label file of N x 3 samples where N is record length in seconds
#The a data is matrix of 21x (NxFs) where fs is 256Hz

# Training Parameters
learning_rate = 0.001
batch_size = 40000
dropout = 0.8

n_recs = np.shape(records)[0]
print(n_recs)
training_epochs = 10
display_epoch = 1

#dfefine logging paths
logs_path = 'S:\\UQCCR-Colditz\\Signal Processing File Sharing\\For Elliot\\Tensorflow_logs\\example2\\'
logs_path = 'C:\\Users\\uqeteo1\\tf_logs\\'
modelPath = 'S:\\UQCCR-Colditz\\Signal Processing File Sharing\\For Elliot\\Tensorflow\\PigletEEG\\model.ckpt'


# Create the neural network
def dense_net(x_dict, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # TF Estimator input is a dict, in case of multiple inputs
        print('network layers Features')
        Feats = x_dict['xFeats']
        print(np.shape(Feats))

        # Output layer, class prediction
        fcl = tf.layers.dense(Feats, 120)
        fcl = tf.layers.dense(fcl, 120)
        fcl = tf.layers.dense(fcl, 40)
        tf.nn.dropout(fcl, keep_prob=dropout)
        out = tf.layers.dense(fcl, 5) #for the 5 classes present in the dataset

        return out

def BiRNN(x, weights, biases):

        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, timesteps, n_input)
        # Required shape: 'timesteps' tensors list of shape (batch_size, num_input)

        # Unstack to get a list of 'timesteps' tensors of shape (batch_size, num_input)
        x = tf.unstack(x, timesteps, 1)

        # Define lstm cells with tensorflow
        # Forward direction cell
        lstm_fw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
        # Backward direction cell
        lstm_bw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

        # Get lstm cell output
        try:
            outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                         dtype=tf.float32)
        except Exception:  # Old TensorFlow version only returns outputs not states
            outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                   dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        return tf.matmul(outputs[-1], weights['out']) + biases['out']

# generate placeholder variables to initiate the network model and training functions the features placeholder is n by features length (dim 2)
xFeats = tf.placeholder(tf.float32, [None, np.shape(Features)[1]], name='InputData')

# the labels placeholder is n by 1
y = tf.placeholder(tf.float32, [None, 5], name='LabelData')



# Construct model and encapsulating all ops into scopes, making
# Tensorboard's Graph visualization more convenient
with tf.name_scope('Model'):
    # Model
    dat = {'xFeats': xFeats}
    logits = conv_net(dat, num_classes, dropout, reuse=False, is_training=True)
    prediction = tf.nn.softmax(logits)
with tf.name_scope('Loss'):
    # Minimize error using cross entropy
    #loss = tf.reduce_mean(tf.pow(logits - y, 2))
    print(prediction)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=y))
with tf.name_scope('SGD'):
    # Gradient Descent
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    # Op to calculate every variable gradient
    grads = tf.gradients(loss, tf.trainable_variables())
    grads = list(zip(grads, tf.trainable_variables()))
    # Op to update all variables according to their gradient
    apply_grads = optimizer.apply_gradients(grads_and_vars=grads)

with tf.name_scope('Accuracy'):
    # Accuracy
    acc,acc_op = tf.metrics.accuracy(labels=y, predictions=prediction)
# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Create a summary to monitor cost tensor
tf.summary.scalar("loss", loss)
# Create a summary to monitor accuracy tensor
tf.summary.scalar("accuracy", tf.squeeze(acc))
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()
saver = tf.train.Saver()
modelAvailFlag = os.path.isfile(modelPath + '.index')
# Start training

with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    # Load the model if it exists
    print('attempting to load model')
    if modelAvailFlag:
        #try restore model
        try:
            saver.restore(sess, modelPath)
            print("Model restored.")
        except:
            print("model is broken, likely the network structure has changed. will overwrite old model")
    else:
        print("model file not present, intialising a new model")

    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    tlast = time.time()
    # Training cycle
    print('starting training')
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_examples/batch_size)+1
        permutation = np.random.permutation(n_examples)
        print('new epoch')
        # Loop over all batches
        for i in range(total_batch):
            print('batch ', i)
            if i < total_batch-1:
                set = permutation[i * batch_size:(i + 1) * batch_size]
                batch_xs = Features[set, :]
                batch_ys = Labels[set]
                print('size of batch , labels', np.shape(batch_xs), np.shape(batch_ys))
            else:
                set = permutation[i * batch_size:]
                batch_xs = Features[set, :]
                batch_ys = Labels[set]
                print('got to last batch')
                print('size of batch , labels', np.shape(batch_xs),  np.shape(batch_ys))


            # Run optimization op (backprop), cost op (to get loss value)
            # and summary nodes
            _, c, summary = sess.run([apply_grads, loss, merged_summary_op], feed_dict={xFeats: batch_xs, y: batch_ys})
            print('batch trained in ', time.time()-tlast, ' seconds, writing to log')
            tlast = time.time()
            # Write logs at every iteration
            summary_writer.add_summary(summary, epoch * total_batch + i)
            # Compute average loss
            print(c)
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch+1) % display_epoch == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
        spath = saver.save(sess, modelPath)
        print('model saved at location:  %s' % spath)
    print("Optimization Finished!")

    # Test model
    # Calculate accuracy
    acc = sess.run(acc_op, feed_dict={xFeats: Features, y: Labels})

    print("Accuracy:", acc)

    print("Run the command line:\n" \
          "--> tensorboard --logdir=C:\\Users\\Tim\\PycharmProjects\\TensorFlow-Examples\\tensorflow_logs\\example\\ " \
          "\nThen open http://0.0.0.0:6006/ into your web browser")
