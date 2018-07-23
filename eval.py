# -*- coding=utf-8 -*-
# author = "tungtt"
import tensorflow as tf
from vn_data_helper import *
from vdcnn import VDCNN
import datetime

import sys


# Model Hyperparameters
tf.flags.DEFINE_integer("sequence_max_length", 1014, "Sequence Max Length (default: 1024)")
tf.flags.DEFINE_string("downsampling_type", "maxpool", "Types of downsampling methods, use either three of maxpool, k-maxpool and linear (default: 'maxpool')")
tf.flags.DEFINE_integer("depth", 9, "Depth for VDCNN, use either 9, 17, 29 or 47 (default: 9)")
tf.flags.DEFINE_boolean("use_he_uniform", True, "Initialize embedding lookup with he_uniform (default: True)")
tf.flags.DEFINE_boolean("optional_shortcut", False, "Use optional shortcut (default: False)")

# Training Parameters
tf.flags.DEFINE_float("learning_rate", 1e-2, "Starter Learning Rate (default: 1e-2)")
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 128)")
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 50)")
tf.flags.DEFINE_integer("evaluate_every", 50, "Evaluate model on dev set after this many steps (default: 50)")
tf.flags.DEFINE_boolean("enable_tensorboard", True, "Enable Tensorboard (default: True)")



FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

data_helper = data_helper(sequence_max_length=FLAGS.sequence_max_length)
train_data, train_label, test_data, test_label = data_helper.load_dataset("dataset/")


cnn = VDCNN(num_classes=train_label.shape[1],
	depth=FLAGS.depth,
	sequence_max_length=FLAGS.sequence_max_length,
	downsampling_type=FLAGS.downsampling_type,
	use_he_uniform=FLAGS.use_he_uniform,
	optional_shortcut=FLAGS.optional_shortcut)


checkpoint_file = tf.train.latest_checkpoint("model/")
print("Checkpoint file"+checkpoint_file)



graph = tf.Graph()
with graph.as_default():
    sess = tf.Session()
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("loss/predictions").outputs[0]
        is_training = graph.get_operation_by_name('is_training').outputs[0]

        # Generate batches for one epoch
        batches = data_helper.batch_iter(list(test_data), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, { input_x: x_test_batch , is_training:False })
            all_predictions = np.concatenate([all_predictions, batch_predictions])
# Print accuracy if y_test is defined
if test_label is not None:
    # correct_predictions = float(sum(all_predictions == test_label))
    y_preds = np.absolute(all_predictions - np.argmax(test_label,axis=1))

    acc = np.count_nonzero(y_preds == 0) / len(y_preds)

    print("Total number of test examples: {}".format(len(test_label)))
    print("Accuracy: {:g}".format(acc))



