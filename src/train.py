from ganpg import *
import ConfigParser as cp
import sys
import os
from datetime import datetime
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
'''
config = tf.ConfigProto()
# use GPU0
config.gpu_options.visible_device_list = '0'
# allocate 50% of GPU memory
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5
session = tf.Session(config=config)
'''
# return format input
def inputProducer(preimgLoc, postimgLoc, batchSize, step, pixelSize=8):
    with open(preimgLoc) as fo:
        pre_lists = np.array(fo.readlines())
    with open(postimgLoc) as fo:
        post_lists = np.array(fo.readlines())

    step = step % (len(pre_lists)/batchSize)
    for i in range(step*batchSize, (step+1)*batchSize):
        pretmp = imread(pre_lists[i][:-1], mode='L')/255
        pretmp = skm.block_reduce(pretmp, (pixelSize, pixelSize), np.average)
        posttmp = imread(post_lists[i][:-1], mode='L')/255
        posttmp = skm.block_reduce(posttmp, (pixelSize, pixelSize), np.average)

        pretmp = np.expand_dims(pretmp, axis=0)
        posttmp = np.expand_dims(posttmp, axis=0)
        if i == step*batchSize:
            prebatch = pretmp
            postbatch = posttmp
        else:
            prebatch = np.concatenate((prebatch, pretmp), axis=0)
            postbatch = np.concatenate((postbatch, posttmp), axis=0)

    prebatch = np.expand_dims(prebatch, axis=3)
    postbatch = np.expand_dims(postbatch, axis=3)

    return prebatch, postbatch

# save image
def sample_img(img_batch, imgpath, step):
    for idx in range(img_batch.shape[0]):
        imsave(os.path.join(imgpath, '%d_%d.png'%(step, idx)), img_batch[idx][:,:,1])

infile = cp.SafeConfigParser()
infile.read(sys.argv[1])

preimg_loc = infile.get('dir', 'preimg_loc')
postimg_loc = infile.get('dir', 'postimg_loc')
save_path = infile.get('dir', 'save_path')
sample_path = infile.get('dir', 'sample_path')
img_size = int(infile.get('feature', 'img_size'))
L1_lambda = float(infile.get('feature', 'L1_lambda'))
batch_size = int(infile.get('feature', 'batch_size'))
lr = float(infile.get('feature', 'learning_rate'))
maxitr = int(infile.get('feature', 'maxitr'))
print_step = int(infile.get('feature', 'print_step'))
sample_step = int(infile.get('feature', 'sample_step'))
save_step = int(infile.get('feature', 'save_step'))
decrease_step = int(infile.get('feature', 'decrease_step'))
leak = float(infile.get('feature', 'leak'))
ckpt = 0
summary_step = int(infile.get('feature', 'summary_step'))

if not os.path.exists(save_path):
    os.mkdir(save_path)
if not os.path.exists(sample_path):
    os.mkdir(sample_path)

# x_AB = tf.concat(preopc, postopc, 3)
x_AB = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 2])
y_real = tf.placeholder(tf.float32, shape=[None, 2])
y_fake = tf.placeholder(tf.float32, shape=[None, 2])
x_realA = x_AB[:, :, :, :1]
x_realB = x_AB[:, :, :, 1:2]

x_fakeB = fwdGeneratorAE(x_realA)

x_realAB = tf.concat([x_realA, x_realB], 3)
x_fakeAB = tf.concat([x_realA, x_fakeB], 3)

d_real = fwdDiscriminator(x_realAB)
d_fake = fwdDiscriminator(x_fakeAB, reuse=True)

g_loss_gan = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=d_fake, labels=y_real))
g_loss_L1 = L1_lambda*tf.reduce_mean(tf.squared_difference(x_fakeB, x_realB))

g_loss = g_loss_gan + g_loss_L1

d_loss_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=d_real, labels=y_real))
d_loss_fake = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=d_fake, labels=y_fake))
d_loss = d_loss_real + d_loss_fake

t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if 'd_' in var.name]
g_vars = [var for var in t_vars if 'g_' in var.name]

dopt = tf.train.RMSPropOptimizer(lr).minimize(d_loss, var_list=d_vars)
gopt = tf.train.RMSPropOptimizer(lr).minimize(g_loss, var_list=g_vars)


with tf.Session() as sess:
    # Send summary statistics to TensorBoard
    tf.summary.scalar('total_Generator_loss', g_loss)
    tf.summary.scalar('Generator_loss_L1',g_loss_L1)
    tf.summary.scalar('Discriminator_loss_real', d_loss_real)
    tf.summary.scalar('Discriminator_loss_fake', d_loss_fake)
    merged = tf.summary.merge_all()
    logdir = "tensorboard/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
    writer = tf.summary.FileWriter(logdir, sess.graph)

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=100)

    # load model
    ckpt = tf.train.get_checkpoint_state(save_path)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        print('[*] RESTORE model: %s' % ckpt_name)
        saver.restore(sess, os.path.join(save_path, ckpt_name))
    else:
        print("[*] LOAD failed!")

    # random label
    _y_real = np.zeros((batch_size, 2), dtype=np.float32)
    _y_fake = np.zeros((batch_size, 2), dtype=np.float32)

    for step in range(maxitr):


        with open(preimg_loc) as fo:
            lists = np.array(fo.readlines())
        randSampling = np.random.randint(len(lists), size=batch_size)


        pre_batch = stochasticInputProducer(preimg_loc, batch_size, randSampling)
        post_batch = stochasticInputProducer(postimg_loc, batch_size, randSampling)
        batch_data = np.concatenate((pre_batch, post_batch), axis=3)

        # radom label
        _y_real[:, 0] = 1.0 - leak + leak * random.random()
        _y_real[:, 1] = 1 - _y_real[:, 0]
        _y_fake[:, 0] = _y_real[:, 1]
        _y_fake[:, 1] = _y_real[:, 0]

        dopt.run(feed_dict={x_AB: batch_data, y_real: _y_real, y_fake: _y_fake})
        gopt.run(feed_dict={x_AB: batch_data, y_real: _y_real, y_fake: _y_fake})

        _d_loss, _d_loss_fake, _d_loss_real, _g_loss, _g_loss_L1 = sess.run(
            [d_loss, d_loss_fake, d_loss_real, g_loss, g_loss_L1],
            feed_dict={x_AB:batch_data, y_real:_y_real, y_fake:_y_fake}
        )

        if step % print_step == 0:
            format_str = ('[%d\%d] %s g_loss: %f, L1 loss: %f, d_loss: %f, d_fake: %f, d_real: %f, lr: %f')
            print(format_str % (step, maxitr, datetime.now(), _g_loss, _g_loss_L1, _d_loss, _d_loss_fake, _d_loss_real, lr))
            print(_y_real[0, 0], _y_real[0, 1])

        if step % sample_step == sample_step - 1:
            print("===========sample======================")
            sample_img(x_fakeAB.eval(feed_dict={x_AB: batch_data}), sample_path, step)
            print('Accuracy: %f'% _g_loss_L1)
            print("=======================================")

        if step % save_step == save_step - 1:
            print("===========save model==================")
            filepath = save_path + 'model-' + str(step) + '-' + str(_g_loss_L1) + '.ckpt'
            saver.save(sess, filepath)
            print("=======================================")

        if step % decrease_step == decrease_step - 1 and lr > 0.000001:
            print("===========decrease lr=================")
            lr /= 10.0
            print("=======================================")

        if step % summary_step == 0:
            # Update TensorBoard with summary statistics
            summary = sess.run(merged,{x_AB: batch_data, y_real: _y_real, y_fake: _y_fake})
            writer.add_summary(summary, step)
