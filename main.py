# This work is based on https://github.com/burliEnterprises/tensorflow-music-generator work.

import numpy as np
import pandas as pd
import msgpack
import glob
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tqdm import tqdm
import os
import midi

# file should be placed in the same director as midi_manipulation.py file and Music_Data directory

import midi_manipulation


def get_songs(path):
    # search inside the path folder and read all files with .mid extension
    files = glob.glob(f'{ path }/*.mid*')
    songs = []
    for f in tqdm(files):
        try:
            song = np.array(midi_manipulation.midiToNoteStateMatrix(f))
            if np.array(song).shape[0] > 50:
                songs.append(song)
        except Exception as e:
            raise e
    return songs

songs = get_songs('Music_Data')  # These songs have already been converted from midi to msgpack
print("{} songs processed".format(len(songs)))

# Model hyperparameters
lowest_note = midi_manipulation.lowerBound # the index of the lowest note on the piano roll
highest_note = midi_manipulation.upperBound # the index of the highest note on the piano roll
note_range = highest_note - lowest_note

# Define training parameters

num_timesteps = 15
n_visible = 2 * note_range * num_timesteps # size of the visible layer
n_hidden = 50 # size of the hidden layer

epochs = 300
batch_size = 100
lr = tf.constant(0.001, tf.float32)

x = tf.compat.v1.placeholder(tf.float32, [None, n_visible], name="x")
W = tf.Variable(tf.random.normal([n_visible, n_hidden], 0.01), name="W")
# There are 2 biases; for the visible bv and hidden bh layers
bh = tf.Variable(tf.zeros([1, n_hidden], tf.float32, name="bh"))
bv = tf.Variable(tf.zeros([1, n_visible], tf.float32, name="bv"))

# Helper function: to sample from a vector of probabilities
def sample(probs):
    # takes in a vector of probabilities, and returns a random vector of 0s and 1s sampled from the input vector
    return tf.floor(probs + tf.random.uniform(tf.shape(probs), 0, 1))

"""
Function below runs the gibb chain. It will be called in two places:
1. When we define the training update step
2. When we sample our music segments from the trained RBM
"""

def gibbs_sample(k):
    # runs a k-step gibbs chain to sample from the probability distribution of the RBM defined by W, bh, bv
    def gibbs_step(count, k, xk):
        # Runs a single gibbs step. The visible values are initialized to xk
        hk = sample(tf.sigmoid(tf.matmul(xk, W) + bh))
        xk = sample(
            # Propagate the hidden values to sample the visible values
            tf.sigmoid(tf.matmul(hk, tf.transpose(W)) + bv)
        )
        return count + 1, k, xk
    
    # Run gibbs steps for k iterations
    ct = tf.constant(0) # counter
    [_, _, x_sample] = control_flow_ops.while_loop(lambda count, num_iter, *args: count<num_iter,
                                                   gibbs_step, [ct, tf.constant(k), x])
    # Not necessary in this implementation except you want to use one of tensorflow's
    # optimizers, you'll need this in order to stop tensorflow from propagating gradients
    # back through the gibbs step
    x_sample = tf.stop_gradient(x_sample)
    return x_sample

"""
Training update code
Implement the contrastive divergence algorithm.
First, get the samples of x and h from the probability distribution
"""
# The sample of x
x_sample = gibbs_sample(1)
# The sample of the hidden nodes, starting from the visible state of x
h = sample(tf.sigmoid(tf.matmul(x, W) + bh))
# The sample of the hidden nodes, starting from the visible state of x_sample
h_sample = sample(tf.sigmoid(tf.matmul(x_sample, W) + bh))

# Next, we update the values of W, bh, and bv based on the difference between the samples that we
# drew and the original values
size_bt = tf.cast(tf.shape(x)[0], tf.float32)
W_adder = tf.multiply(lr / size_bt,
                      tf.subtract(tf.matmul(tf.transpose(x), h), tf.matmul(tf.transpose(x_sample), h_sample)))
bv_adder = tf.multiply(lr / size_bt, tf.reduce_sum(tf.subtract(x, x_sample), 0, True))
bh_adder = tf.multiply(lr / size_bt, tf.reduce_sum(tf.subtract(h, h_sample), 0, True))
# When we run sess.run(updt), Tensorflow will run all 3 update steps
updt = [W.assign_add(W_adder), bv.assign_add(bv_adder), bh.assign_add(bh_adder)]

# Run the graph!
# Now it's time to start a session and run the graph!
with tf.Session() as sess:
    # First, we train the model
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)
    # Run through all of the training data epochs times
    for epoch in tqdm(range(epochs)):
        for song in songs:
            # The songs are stored in a time x notes format. The size of each song is timesteps_in_song x 2*note_range
            # Here we reshape the songs so that each training example
            # is a vector with num_timesteps x 2*note_range elements
            song = np.array(song)
            song = song[:int(np.floor(song.shape[0] // num_timesteps) * num_timesteps)]
            song = np.reshape(song, [song.shape[0] // num_timesteps, song.shape[1] * num_timesteps])
            # Train the RBM on batch_size examples at a time
            for i in range(1, len(song), batch_size):
                tr_x = song[i:i + batch_size]
                sess.run(updt, feed_dict={x: tr_x})
                
    # Now the model is fully trained, so let's make some music!
    # Run a gibbs chain where the visible nodes are initialized to 0
    sample = gibbs_sample(1).eval(session=sess, feed_dict={x: np.zeros((10, n_visible))})
    for i in range(sample.shape[0]):
        if not any(sample[i, :]):
            continue
        # Here we reshape the vector to be time x notes and then save the vector as a midi file
        S = np.reshape(sample[i, :], (num_timesteps, 2 * note_range))
        midi_manipulation.noteStateMatrixToMidi(S, f"out/generated_chord_{i}")
    print("done training. outputs saved in the out folder")
