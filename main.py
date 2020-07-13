'''
@author: Wallace Lira
Our GANhopper implementation was based on xhujoy's tensorflow CycleGAN repository, available at https://github.com/xhujoy/CycleGAN-tensorflow 
'''

import argparse
import os
import tensorflow as tf
from model import cyclegan

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_dir', dest='dataset_dir', default='cat_dog_face', help='path of the dataset')
parser.add_argument('--epoch', dest='epoch', type=int, default=100, help='# of epoch')
parser.add_argument('--epoch_step', dest='epoch_step', type=int, default=75, help='# of epoch to decay lr')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=6, help='# images in batch')
parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='# images used to train')
parser.add_argument('--load_size', dest='load_size', type=int, default=146, help='scale images to this size')
parser.add_argument('--fine_size', dest='fine_size', type=int, default=128, help='then crop to this size')
parser.add_argument('--ngf', dest='ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=3, help='# of input image channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=3, help='# of output image channels')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--which_direction', dest='which_direction', default='AtoB', help='AtoB or BtoA')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--save_freq', dest='save_freq', type=int, default=1000, help='save a model every save_freq iterations')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=100, help='print the debug information every print_freq iterations')
parser.add_argument('--continue_train', dest='continue_train', type=bool, default=True, help='if continue training, load the latest model: 1: true, 0: false')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--use_resnet', dest='use_resnet', type=bool, default=True, help='generation network using reidule block')
parser.add_argument('--use_lsgan', dest='use_lsgan', type=bool, default=True, help='gan loss defined in lsgan')
parser.add_argument('--max_size', dest='max_size', type=int, default=50, help='max size of image pool, 0 means do not use image pool')
parser.add_argument('--gpu', dest='gpu', type=int, default=1, help='choose gpu')

#objective function parameters
parser.add_argument('--hybridness', dest='hybridness', type=float, default=1.0, help='weight of the hybridness loss')
parser.add_argument('--adversarial', dest='adversarial', type=float, default=1.0, help='weight of the adversarial loss')
parser.add_argument('--h_hops', dest='h_hops', type=int, default=4, help='total number of translation hops between two domains')
parser.add_argument('--smootheness', dest='smootheness', type=float, default=2.5, help='weight of the smootheness loss')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=10.0, help='weight on reconstruction loss term between hops in objective')

args = parser.parse_args()

def main(_):
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    smoothenes_seperated = str(args.smootheness).split('.')
    model_dir = "%s_%s_%s_%s_%s" % (args.dataset_dir, args.fine_size, args.h_hops, smoothenes_seperated[0], smoothenes_seperated[1])
    if not os.path.exists(args.sample_dir+"/"+model_dir):
        os.makedirs(args.sample_dir+"/"+model_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)

    with tf.device('/gpu:{}'.format(args.gpu)):
        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth = True
        with tf.Session(config=tfconfig) as sess:
            model = cyclegan(sess, args)
            model.train(args) if args.phase == 'train' \
                else model.test(args)

if __name__ == '__main__':
    tf.app.run()
