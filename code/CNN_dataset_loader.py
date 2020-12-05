from os.path import join
import numpy as np
from CNN_const import *

class DatasetLoader(object):

  def __init__(self):
    pass

  def load_from_save(self):

    self._images = np.load(join(SAVE_DIRECTORY_DATA, SAVE_DATASET_IMAGES_FILENAME))
    self._labels = np.load(join(SAVE_DIRECTORY_DATA, SAVE_DATASET_LABELS_FILENAME))
    sz=self._images.shape[0]
    train_lbl = np.zeros((sz, len(EMOTIONS)))
    for i in range(sz):
      if self._labels[i]==1:
        train_lbl[i, 0]=1
      else:
        train_lbl[i, 1]=1

    self._labels = train_lbl

    self._images_valid = np.load(join(SAVE_DIRECTORY_DATA, SAVE_DATASET_IMAGES_VALID_FILENAME))
    self._labels_valid = np.load(join(SAVE_DIRECTORY_DATA, SAVE_DATASET_LABELS_VALID_FILENAME))
    sz = self._images_valid.shape[0]
    val_lbl = np.zeros((sz, len(EMOTIONS)))
    for i in range(sz):
      if self._labels_valid[i] == 1:
        val_lbl[i, 0] = 1
      else:
        val_lbl[i, 1] = 1

    self._labels_valid = val_lbl

    self._images_test = np.load(join(SAVE_DIRECTORY_DATA, SAVE_DATASET_IMAGES_TEST_FILENAME))
    self._labels_test = np.load(join(SAVE_DIRECTORY_DATA, SAVE_DATASET_LABELS_TEST_FILENAME))
    sz = self._images_test.shape[0]
    test_lbl = np.zeros((sz, len(EMOTIONS)))
    for i in range(sz):
      if self._labels_test[i] == 1:
        test_lbl[i, 0] = 1
      else:
        test_lbl[i, 1] = 1

    self._labels_test = test_lbl


    self._images = self._images.reshape([-1, SIZE_FACE, SIZE_FACE, 1])
    self._labels = self._labels.reshape([-1, len(EMOTIONS)])

    self._images_valid = self._images_valid.reshape([-1, SIZE_FACE, SIZE_FACE, 1])
    self._labels_valid = self._labels_valid.reshape([-1, len(EMOTIONS)])

    self._images_test = self._images_test.reshape([-1, SIZE_FACE, SIZE_FACE, 1])
    self._labels_test = self._labels_test.reshape([-1, len(EMOTIONS)])


  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def images_test(self):
    return self._images_test

  @property
  def labels_test(self):
    return self._labels_test

  @property
  def images_valid(self):
    return self._images_valid

  @property
  def labels_valid(self):
    return self._labels_valid

  @property
  def num_examples(self):
    return self._num_examples
