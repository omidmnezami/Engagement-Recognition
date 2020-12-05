from __future__ import division, absolute_import
import numpy as np
from CNN_dataset_loader import DatasetLoader
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from CNN_const import *
from os.path import isfile, join
import sys


class EmotionRecognition:

  def __init__(self):
    self.dataset = DatasetLoader()
    # self.lr =lr
  def build_network(self):
    img_aug = tflearn.ImageAugmentation()
    img_aug.add_random_flip_leftright()
    # img_aug.add_random_flip_updown()
    img_aug.add_random_crop([SIZE_FACE, SIZE_FACE], padding=4)
    img_aug.add_random_rotation(max_angle=8.0)
    
    img_prep = tflearn.ImagePreprocessing()
    img_prep.add_featurewise_zero_center(per_channel=True)
    img_prep.add_featurewise_stdnorm(per_channel=True)

    print('[+] Building CNN')
    self.network = input_data(shape = [None, SIZE_FACE, SIZE_FACE, 1], data_preprocessing=img_prep, data_augmentation=img_aug)

    self.network = conv_2d(self.network, 64, 5, activation='relu')
    self.network = local_response_normalization(self.network)
    self.network = max_pool_2d(self.network, 3, strides=2)
    self.network = dropout(self.network, 0.8)

    self.network = conv_2d(self.network, 128, 3, activation='relu')
    self.network = max_pool_2d(self.network, 3, strides=2)
    self.network = dropout(self.network, 0.8)

    self.network = fully_connected(self.network, 1024, activation='relu', weight_decay=0.0001)
    self.network = dropout(self.network, 0.7)

    self.network = fully_connected(self.network, 1024, activation='relu', weight_decay=0.0001)
    self.network = dropout(self.network, 0.7)

    self.network = fully_connected(self.network, len(EMOTIONS), activation='softmax')

    mom = tflearn.optimizers.Momentum(learning_rate=0.02, lr_decay=0.8, decay_step=500)

    self.network = regression(self.network, optimizer=mom, loss='categorical_crossentropy')

    self.model = tflearn.DNN(
      self.network,
      tensorboard_dir = './tmp/',
      checkpoint_path = None,
      max_checkpoints = None,
      tensorboard_verbose = 0
    )

  def load_saved_dataset(self):
    self.dataset.load_from_save()
    print('[+] Dataset found and loaded')

  def start_training(self):
    self.load_saved_dataset()
    self.build_network()
    if self.dataset is None:
      self.load_saved_dataset()
    # Training
    print('[+] Training network')
    epoch_num = 20
    # early_stopping_cb = StoppingCallback(self, val_acc_arr=np.zeros(epoch_num))
    self.model.fit(
      self.dataset.images, self.dataset.labels,
      validation_set = (self.dataset.images_valid, self.dataset._labels_valid),
      n_epoch = epoch_num,
      batch_size = 28,
      shuffle = True,
      show_metric = True,
      snapshot_step = None,
      snapshot_epoch = True,
      run_id = 'emotion_recognition'
      # callbacks = early_stopping_cb
    )

  def predict(self, image):
    if image is None:
      return None
    image = image.reshape([-1, SIZE_FACE, SIZE_FACE, 1])
    return self.model.predict(image)

  def save_model(self):
    self.model.save(join(SAVE_DIRECTORY, SAVE_MODEL_FILENAME))
    print('[+] Model trained and saved at ' + SAVE_MODEL_FILENAME)

  def load_model(self):
    if isfile(join(SAVE_DIRECTORY, SAVE_MODEL_FILENAME+'.index')):
      self.model.load(join(SAVE_DIRECTORY, SAVE_MODEL_FILENAME))
      print('[+] Model ;loaded from ' + SAVE_MODEL_FILENAME)
      print '********* The test set accuracy is: '
      perfval  = self.model.evaluate(self.dataset.images_valid, self.dataset._labels_valid)
      perftest = self.model.evaluate(self.dataset.images_test, self.dataset.labels_test)

      print perfval, perftest

      predictval = self.model.predict(self.dataset.images_valid)
      predicttest = self.model.predict(self.dataset.images_test)

      np.save('CNNval_f.npy',predictval)
      np.save( 'CNNtest_f.npy', predicttest)

      return perftest
    else:
      print '[-] The model is not found'
      print '[-] The model is not found'
      print '[-] The model is not found'
      return 0.0

    # if isfile(join(SAVE_DIRECTORY, SAVE_MODEL_FILENAME+'.index')):
    #   self.model.load(join(SAVE_DIRECTORY, SAVE_MODEL_FILENAME))
    #   print('[+] Model loaded from ' + SAVE_MODEL_FILENAME)
    #   print '********* The test set accuracy is: '
    #   print self.model.evaluate(self.dataset.images_test, self.dataset.labels_test)
    # else:
    #   print '[-] Model is not found'

def show_usage():
  # I din't want to have more dependecies
  print('You can select train or test as input')

if __name__ == "__main__":

  network = EmotionRecognition()
  if sys.argv[1] == 'train':
    network.start_training()
    network.save_model()
    network.load_model()
  elif sys.argv[1] == 'test':
    network.load_saved_dataset()
    network.build_network()
    network.load_model()
  else:
    show_usage()
