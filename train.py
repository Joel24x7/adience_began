import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from data_prep import get_list_from_h5py, save_to_h5py
from model import Began

data_name = 'celeb'
project_num = 1.8

def train(model, epochs=100):

    np.random.RandomState(123)
    tf.set_random_seed(123)

    #Setup file structure
    project_dir, logs_dir, samples_dir, models_dir = setup_dirs(project_num)
    checkpoint_root = tf.train.latest_checkpoint(models_dir, latest_filename=None)
    if checkpoint_root != None:
        tf.reset_default_graph()

    if not os.path.exists("{}.h5".format(data_name)):
        raise Exception('Data unavailable. Run data_prep.py first')

    #Hyperparameters
    inital_lr = 0.00008
    lambda_kt = 0.001
    gamma = 0.5
    sess_kt = 0.0
    epoch_drop = 350

    #Global Step
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(-1), trainable=False)
    increment_step = global_step.assign(global_step + 1.0)

    #Setup model
    x, z, lr, kt = model.initInputs()
    dis_loss, gen_loss, d_x_loss, d_z_loss = model.loss(x, z, kt)
    dis_opt, gen_opt = model.optimizer(dis_loss, gen_loss, lr)
    m_global = d_x_loss + tf.abs(gamma * d_x_loss - d_z_loss)

    #For saving sample images during training or in test
    start_time = time.time()
    sample = model.get_sample(3)

    #Setup data 
    data = get_list_from_h5py(data_name)
    batch_size = model.batch_size
    num_batches_per_epoch = len(data) // batch_size

    #Tensorboard
    tf.summary.scalar('convergence', m_global)
    tf.summary.scalar('kt', kt)
    merged = tf.summary.merge_all()

    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    # config.log_device_placement=True
    with tf.Session(config=config) as sess:
        train_writer = tf.summary.FileWriter('./{}'.format(logs_dir), sess.graph)

        #Load previous training
        if checkpoint_root != None:
            saver.restore(sess, checkpoint_root)
            print('\nRestored Previous Training\n')
            print("Starting at Global Step: {}".format(global_step.eval()))
        else:
            sess.run(init_op)
            print("Starting at Global Step: {}".format(global_step.eval()))

        
        for epoch in range(epochs):

            np.random.shuffle(data)
            learning_rate = inital_lr * math.pow(0.5, epoch+1 // epoch_drop)

            for batch_step in range(num_batches_per_epoch):

                #Prep batch
                start_data_batch = batch_step * batch_size
                end_data_batch = start_data_batch + batch_size
                batch_data = data[start_data_batch:end_data_batch, :, :, :]
                z_batch = np.random.uniform(-1, 1, size=[batch_size, model.noise_dim])

                fetches = [dis_opt, gen_opt, d_x_loss, d_z_loss, increment_step]
                feed_dict={x: batch_data, z: z_batch, lr: learning_rate, kt: sess_kt}
                _, _, real_loss, fake_loss, int_step = sess.run(fetches=fetches, feed_dict=feed_dict)

                balance = gamma * real_loss - fake_loss
                sess_kt = np.minimum(1.0, np.maximum(sess_kt + lambda_kt * (balance), 0.0))
                convergence = real_loss + np.abs(balance)

                print('Time: {} Epoch: {} Global Step: {} - {}/{} convergence: {:.4} kt: {:.4}'.format(int(time.time() - start_time), epoch, int_step, batch_step, num_batches_per_epoch, convergence, sess_kt))

                if int_step % 300 == 0:
                    summary = sess.run(merged, feed_dict)
                    train_writer.add_summary(summary, int_step)
                    saver.save(sess, './{}/began'.format(models_dir), global_step=global_step)

                    images = sess.run(sample)
                    for i in range(images.shape[0]):
                        tmp_name = '{}/train_{}_{}.png'.format(samples_dir, int_step, i)
                        img = images[i, :, :, :]
                        plt.imshow(img)
                        plt.savefig(tmp_name)

                        # Uncomment to see training images too
                        # x_name = '{}/data_{}_{}.png'.format(samples_d ir, curr_step, i)
                        # data_img = batch_data[i, :, :, :]
                        # plt.imshow(data_img)
                        # plt.savefig(x_name)

        summary = sess.run(merged, feed_dict)
        train_writer.add_summary(summary, global_step.eval())
        saver.save(sess, './{}/began'.format(models_dir), global_step=global_step)
        train_writer.close()

def test(model):

    #Setup file structure
    project_dir, logs_dir, samples_dir, models_dir = setup_dirs(project_num)

    #Setup model
    checkpoint_root = tf.train.latest_checkpoint(models_dir,latest_filename=None)
    if checkpoint_root != None:
        tf.reset_default_graph()

    sample = model.get_sample(7, reuse=False)
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:

        if checkpoint_root != None:
            saver.restore(sess, checkpoint_root)
            print('\nRestored Previous Training\n')

        else:
            sess.run(init_op)


        images = sess.run(sample)
        for i in range(images.shape[0]):
            tmpName = '{}/test_image{}.png'.format(samples_dir, i)
            img = images[i, :, :, :]
            plt.imshow(img)
            plt.savefig(tmpName)

def setup_dirs(project_num):

    project_dir = '{}_began_{}'.format(data_name, project_num)
    logs_dir = '{}/logs_{}'.format(project_dir, project_num)
    samples_dir = '{}/results_{}'.format(project_dir, project_num)
    models_dir = '{}/models_{}'.format(project_dir, project_num)

    if not os.path.exists(project_dir):
        os.makedirs(project_dir)
        os.makedirs(logs_dir)
        os.makedirs(samples_dir)
        os.makedirs(models_dir)

    return project_dir, logs_dir, samples_dir, models_dir
