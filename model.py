from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from collections import namedtuple

from module import *
from utils import *


class cyclegan(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.batch_size = args.batch_size
        self.image_size = args.fine_size
        self.input_c_dim = args.input_nc
        self.output_c_dim = args.output_nc
        self.L1_lambda = args.L1_lambda
        self.dataset_dir = args.dataset_dir
        self.h_hops = args.h_hops
        self.load_size = args.load_size
        self.fine_size = args.fine_size
        self.smootheness = args.smootheness
        self.adversarial = args.adversarial
        self.hybridness = args.hybridness
        self.smoothenes_seperated = str(self.smootheness).split('.')

        self.discriminator = discriminator
        if args.use_resnet:
            self.generator = generator_resnet
        else:
            self.generator = generator_unet
        if args.use_lsgan:
            self.criterionGAN = mae_criterion
        else:
            self.criterionGAN = sce_criterion

        OPTIONS = namedtuple('OPTIONS', 'batch_size image_size \
                              gf_dim df_dim output_c_dim is_training')
        self.options = OPTIONS._make((args.batch_size, args.fine_size,
                                      args.ngf, args.ndf, args.output_nc,
                                      args.phase == 'train'))

        self._build_model()
        self.saver = tf.train.Saver()
        self.pool = ImagePool(args.max_size)

    def _build_model(self):
        self.real_data = tf.placeholder(tf.float32,
                                        [None, self.image_size, self.image_size,
                                         self.input_c_dim + self.output_c_dim],
                                        name='real_A_and_B_images')

        self.interpolation_rate = tf.placeholder(tf.float32, shape=(), name='interpolation_rate')
        self.n_baskets = tf.placeholder(tf.float32, shape=(), name='n_baskets')

        self.real_A = self.real_data[:, :, :, :self.input_c_dim]
        self.real_B = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]

        self.fake_B = self.generator(self.real_A, self.options, False, name="generatorA2B")
        self.fake_A_ = self.generator(self.fake_B, self.options, False, name="generatorB2A")
        self.fake_A = self.generator(self.real_B, self.options, True, name="generatorB2A")
        self.fake_B_ = self.generator(self.fake_A, self.options, True, name="generatorA2B")

        self.DB_fake = self.discriminator(self.fake_B, self.options, reuse=False, name="discriminatorB")
        self.DA_fake = self.discriminator(self.fake_A, self.options, reuse=False, name="discriminatorA")
	
        self.DA_fake_classifier = self.discriminator(self.fake_A, self.options, reuse=False, name="discriminatorClassifier")
        self.DB_fake_classifier = self.discriminator(self.fake_B, self.options, reuse=True, name="discriminatorClassifier")
        self.DB_classifier = self.discriminator(self.real_B, self.options, reuse=True, name="discriminatorClassifier")
        self.DA_classifier = self.discriminator(self.real_A, self.options, reuse=True, name="discriminatorClassifier")
	
        self.g_loss_a2b = \
            + self.adversarial * self.criterionGAN(self.DB_fake, tf.ones_like(self.DB_fake)) \
            + self.hybridness * self.criterionGAN(self.DB_fake_classifier, tf.scalar_mul(self.interpolation_rate, tf.ones_like(self.DB_classifier))) \
            + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
            + (self.smootheness) * abs_criterion(self.real_A, self.fake_B)
        self.g_loss_b2a = \
            self.adversarial * self.criterionGAN(self.DA_fake, tf.ones_like(self.DA_fake)) \
            + self.hybridness * self.criterionGAN(self.DA_fake_classifier, tf.scalar_mul(1. - self.interpolation_rate, tf.ones_like(self.DA_classifier))) \
            + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_) \
            + (self.smootheness) * abs_criterion(self.real_B, self.fake_A)
        self.g_loss = \
            self.adversarial * self.criterionGAN(self.DA_fake, tf.ones_like(self.DA_fake)) \
            + self.adversarial * self.criterionGAN(self.DB_fake, tf.ones_like(self.DB_fake)) \
            + self.hybridness * self.criterionGAN(self.DA_fake_classifier, tf.scalar_mul(1. - self.interpolation_rate, tf.ones_like(self.DA_classifier))) \
            + self.hybridness * self.criterionGAN(self.DB_fake_classifier, tf.scalar_mul(self.interpolation_rate, tf.ones_like(self.DB_classifier))) \
            + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
            + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_) \
            + (self.smootheness) * abs_criterion(self.real_A, self.fake_B) \
            + (self.smootheness) * abs_criterion(self.real_B, self.fake_A)



        self.fake_A_sample = tf.placeholder(tf.float32,
                                            [None, self.image_size, self.image_size,
                                             self.input_c_dim], name='fake_A_sample')
        self.fake_B_sample = tf.placeholder(tf.float32,
                                            [None, self.image_size, self.image_size,
                                             self.output_c_dim], name='fake_B_sample')


        #classifier losses
        self.da_loss_classifier = self.criterionGAN(self.DA_classifier, tf.zeros_like(self.DA_classifier))
        self.db_loss_classifier = self.criterionGAN(self.DB_classifier, tf.ones_like(self.DB_classifier))
        self.d_loss_classifier = (self.da_loss_classifier + self.db_loss_classifier) / 2

        self.DB_real = self.discriminator(self.real_B, self.options, reuse=True, name="discriminatorB")
        self.DA_real = self.discriminator(self.real_A, self.options, reuse=True, name="discriminatorA")
        self.DB_fake_sample = self.discriminator(self.fake_B_sample, self.options, reuse=True, name="discriminatorB")
        self.DA_fake_sample = self.discriminator(self.fake_A_sample, self.options, reuse=True, name="discriminatorA")

        self.db_loss_real = self.criterionGAN(self.DB_real, tf.ones_like(self.DB_real))
        self.db_loss_fake = self.criterionGAN(self.DB_fake_sample, tf.zeros_like(self.DB_fake_sample))
        self.db_loss = (self.db_loss_real + self.db_loss_fake) / 2
        self.da_loss_real = self.criterionGAN(self.DA_real, tf.ones_like(self.DA_real))
        self.da_loss_fake = self.criterionGAN(self.DA_fake_sample, tf.zeros_like(self.DA_fake_sample))
        self.da_loss = (self.da_loss_real + self.da_loss_fake) / 2
        self.d_loss = self.da_loss + self.db_loss + self.d_loss_classifier

        self.g_loss_a2b_sum = tf.summary.scalar("g_loss_a2b", self.g_loss_a2b)
        self.g_loss_b2a_sum = tf.summary.scalar("g_loss_b2a", self.g_loss_b2a)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.g_sum = tf.summary.merge([self.g_loss_a2b_sum, self.g_loss_b2a_sum, self.g_loss_sum])
        self.db_loss_sum = tf.summary.scalar("db_loss", self.db_loss)
        self.da_loss_sum = tf.summary.scalar("da_loss", self.da_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.db_loss_real_sum = tf.summary.scalar("db_loss_real", self.db_loss_real)
        self.db_loss_fake_sum = tf.summary.scalar("db_loss_fake", self.db_loss_fake)
        self.da_loss_real_sum = tf.summary.scalar("da_loss_real", self.da_loss_real)
        self.da_loss_fake_sum = tf.summary.scalar("da_loss_fake", self.da_loss_fake)
        self.d_sum = tf.summary.merge(
            [self.da_loss_sum, self.da_loss_real_sum, self.da_loss_fake_sum,
             self.db_loss_sum, self.db_loss_real_sum, self.db_loss_fake_sum,
             self.d_loss_sum]
        )

        self.test_A = tf.placeholder(tf.float32,
                                     [None, self.image_size, self.image_size,
                                      self.input_c_dim], name='test_A')
        self.test_B = tf.placeholder(tf.float32,
                                     [None, self.image_size, self.image_size,
                                      self.output_c_dim], name='test_B')
        self.testB = self.generator(self.test_A, self.options, True, name="generatorA2B")
        self.testA = self.generator(self.test_B, self.options, True, name="generatorB2A")

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]
        for var in t_vars: print(var.name)

    def train(self, args):
        """Train cyclegan"""
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        self.d_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        counter = 1
        start_time = time.time()

        if args.continue_train:
            if self.load(args.checkpoint_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

        #create interpolation baskets
        interpolation_rates = []
        n_baskets = self.h_hops
        step = 1/(n_baskets)
        for i in range(n_baskets):
            interpolation_rates.append(i*step + step)

        for epoch in range(args.epoch):

            dataA = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainA'))
            dataB = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainB'))
            np.random.shuffle(dataA)
            np.random.shuffle(dataB)
            batch_idxs = min(min(len(dataA), len(dataB)), args.train_size) // self.batch_size
            lr = args.lr if epoch < args.epoch_step else args.lr*(args.epoch-epoch)/(args.epoch-args.epoch_step)


            for idx in range(0, batch_idxs):

                batch_files = list(zip(dataA[idx * self.batch_size:(idx + 1) * self.batch_size],
                                       dataB[idx * self.batch_size:(idx + 1) * self.batch_size]))
                batch_images = [load_train_data(batch_file, args.load_size, args.fine_size) for batch_file in batch_files]
                batch_images = np.array(batch_images).astype(np.float32)

                current_input = batch_images

                #update generator and discriminator network for each basket
                for k, interpolation_rate in enumerate(interpolation_rates):
                    # Update G network and record fake outputs for each intermediary basket
                    fake_A, fake_B, _, summary_str = self.sess.run(
                        [self.fake_A, self.fake_B, self.g_optim, self.g_sum],
                        feed_dict={self.real_data: current_input,
                                   self.lr: lr,
                                   self.interpolation_rate: interpolation_rate,
                                   self.n_baskets: n_baskets})
                    self.writer.add_summary(summary_str, counter)
                    [fake_A, fake_B] = self.pool([fake_A, fake_B])

                    #create new input for the next iteration of generator training
                    current_input = np.concatenate((fake_B, fake_A), 3)

                    # Update D network using exclusively real samples
                    _, summary_str = self.sess.run(
                        [self.d_optim, self.d_sum],
                        feed_dict={self.real_data: batch_images,
                                   self.fake_A_sample: fake_A,
                                   self.fake_B_sample: fake_B,
                                   self.lr: lr,
                                   self.interpolation_rate: interpolation_rate,
                                   self.n_baskets: n_baskets})
                    self.writer.add_summary(summary_str, counter)

                    if np.mod(counter, args.print_freq) == 1:
                        self.sample_model(args.sample_dir, epoch, idx)

                    if np.mod(counter, args.save_freq) == 2:
                        self.save(args.checkpoint_dir, counter)

                counter += 1
                print(("Epoch: [%2d] [%4d/%4d] time: %4.4f" % (
                    epoch, idx, batch_idxs, time.time() - start_time)))
					
    def save(self, checkpoint_dir, step):
        model_name = "cyclegan.model"
        model_dir = "%s_%s_%s_%s_%s" % (self.dataset_dir, self.image_size, self.h_hops, self.smoothenes_seperated[0], self.smoothenes_seperated[1])
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s_%s_%s_%s" % (self.dataset_dir, self.image_size, self.h_hops, self.smoothenes_seperated[0], self.smoothenes_seperated[1])
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def sample_model(self, sample_dir, epoch, idx):
        dataA = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testA'))
        dataB = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testB'))
        np.random.shuffle(dataA)
        np.random.shuffle(dataB)
        batch_files = list(zip(dataA[:self.batch_size], dataB[:self.batch_size]))
        sample_images = [load_train_data(batch_file, self.load_size, self.fine_size, is_testing=True) for batch_file in batch_files]
        sample_images = np.array(sample_images).astype(np.float32)

        model_dir = "%s_%s_%s_%s_%s" % (self.dataset_dir, self.image_size, self.h_hops, self.smoothenes_seperated[0], self.smoothenes_seperated[1])

        save_images(sample_images[:, :, :, :3], [self.batch_size, 1],
                    './{}/{}/B_{:02d}_{:04d}_{:02d}.jpg'.format(sample_dir, model_dir, epoch, idx, 0))
        save_images(sample_images[:, :, :, 3:], [self.batch_size, 1],
                    './{}/{}/A_{:02d}_{:04d}_{:02d}.jpg'.format(sample_dir, model_dir, epoch, idx, 0))

        for i in range((self.h_hops) * 2):
            fake_A, fake_B = self.sess.run(
                [self.fake_A, self.fake_B],
                feed_dict={self.real_data: sample_images}
            )
            sample_images = np.concatenate((fake_B, fake_A), 3)
            save_images(fake_A, [self.batch_size, 1],
                        './{}/{}/A_{:02d}_{:04d}_{:02d}.jpg'.format(sample_dir, model_dir, epoch, idx, i+1))
            save_images(fake_B, [self.batch_size, 1],
                        './{}/{}/B_{:02d}_{:04d}_{:02d}.jpg'.format(sample_dir, model_dir, epoch, idx, i+1))

    def ensure_dir(self, file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
            return file_path
        else:
            return file_path

    def test(self, args):
        """Test cyclegan"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        if args.which_direction == 'AtoB':
            sample_files = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testA'))
        elif args.which_direction == 'BtoA':
            sample_files = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testB'))
        else:
            raise Exception('--which_direction must be AtoB or BtoA')

        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

  
        model_dir = "%s_%s_%s_%s_%s" % (self.dataset_dir, self.image_size, self.h_hops, self.smoothenes_seperated[0], self.smoothenes_seperated[1])

        # write html for visual comparison
        index_path = os.path.join(args.test_dir, '{0}_{1}_index.html'.format(model_dir,args.which_direction))
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")

        #creat columns
        columns = ""
        for i in range((self.h_hops) * 3):
            if i < self.h_hops:
                title = "interpolation"
                number = i + 1
            else:
                title = "extrapolation"
                number = i - self.h_hops
            columns+="<th>"+title+" "+str(number)+"</th>"


        index.write("<th>name</th><th>input</th>"+columns+"</tr>")

        out_var, in_var = (self.testB, self.test_A) if args.which_direction == 'AtoB' else (
            self.testA, self.test_B)

        for sample_file in sample_files:
            print('Processing image: ' + sample_file)
            sample_image = [load_test_data(sample_file, args.fine_size)]
            sample_image = np.array(sample_image).astype(np.float32)


            image_path_original = os.path.join(args.test_dir, model_dir)
            if not os.path.exists(image_path_original):
                os.makedirs(image_path_original)

            file_name = os.path.join(image_path_original,
                                     '{0}_{1}_{2}'.format(args.which_direction, str(0).zfill(2),
                                                          os.path.basename(sample_file)))
            save_images(sample_image, [1, 1], file_name)

            index.write("<td>%s</td>" % os.path.basename(file_name))
            index.write("<td><img src='%s'></td>" % (file_name if os.path.isabs(file_name) else ('..' + os.path.sep + file_name)))
            for i in range((self.h_hops) * 3):
                fake_img = self.sess.run(out_var, feed_dict={in_var: sample_image})
                sample_image = fake_img

                file_name = os.path.join(image_path_original,
                                          '{0}_{1}_{2}'.format(args.which_direction, str(i+1).zfill(2) ,os.path.basename(sample_file)))

                save_images(fake_img, [1, 1], file_name)

                index.write("<td><img src='%s'></td>" % (file_name if os.path.isabs(file_name) else (
                    '..' + os.path.sep + file_name)))
            index.write("</tr>")



        index.close()
        index.close()
