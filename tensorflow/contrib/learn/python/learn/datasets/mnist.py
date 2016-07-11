# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Functions for downloading and reading MNIST data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip

import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'


def _read32(bytestream):
  dt = numpy.dtype(numpy.uint32).newbyteorder('>') # uint32의 big endian 방식의 데이터타입이다.
  return numpy.frombuffer(bytestream.read(4), dtype=dt)[0] #4바이트를 빅 인디안 방식으로 읽어서 리턴 시켜준다.


def extract_images(f):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth].

  Args:
    f: A file object that can be passed into a gzip reader.

  Returns:
    data: A 4D unit8 numpy array [index, y, x, depth].

  Raises:
    ValueError: If the bytestream does not start with 2051.

  """
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2051: #magic 값이 2051이 아니면 정상적인이미지 파이들의 집합이 아니다.
      raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                       (magic, f.name))
    num_images = _read32(bytestream) #이미지 갯수
    rows = _read32(bytestream) #이미지의 넓이 
    cols = _read32(bytestream) #이미지의 높이
    buf = bytestream.read(rows * cols * num_images) # 전체 이미지의 크기를 읽어서 buf에 넣는다.
    data = numpy.frombuffer(buf, dtype=numpy.uint8) #uint8  type으로 buf의 다중 배열을 생성한다.
    data = data.reshape(num_images, rows, cols, 1) #배열을 (갯수, 행, 열, 1)형태의 배열로 만들어준다. 1은 왜 넣었을까요??
    return data


def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0] # 60,000개의 라벨이 들어가 있다.
  index_offset = numpy.arange(num_labels) * num_classes #[0, 10, 20, .... 599,990] 배열로 값이 들어간다.
  labels_one_hot = numpy.zeros((num_labels, num_classes)) #0의로 채워진 60,000, 10의 2차 배열을 만들어준다.
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


def extract_labels(f, one_hot=False, num_classes=10):
  """Extract the labels into a 1D uint8 numpy array [index].

  Args:
    f: A file object that can be passed into a gzip reader.
    one_hot: Does one hot encoding for the result.
    num_classes: Number of classes for the one hot encoding.

  Returns:
    labels: a 1D unit8 numpy array.

  Raises:
    ValueError: If the bystream doesn't start with 2049.
  """
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream) # 32바이트를 읽고
    if magic != 2049: #2049가 아니면MNIST label이 아니다.
      raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                       (magic, f.name))
    num_items = _read32(bytestream) #item 갯수
    buf = bytestream.read(num_items) #아이템 갯수만큼 byte갯수를 읽어온다.
    labels = numpy.frombuffer(buf, dtype=numpy.uint8) #uint8 타입으로 다중 어레이를 만든다.
    if one_hot:
      return dense_to_one_hot(labels, num_classes)
    return labels


class DataSet(object):

  def __init__(self,
               images,
               labels,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32,
               reshape=True):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.
    """
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):#dtype 값 검증 uint8, float32 값이 아니라면 아래 타입에러를 띄운다.
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape)) #images.shape와  labels.shape갯수가 다르면 메시지를 뿌리고 끝낸다.
      self._num_examples = images.shape[0]  #_num_examples 값에 이미지 갯수를 넣어준다.

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      if reshape:
        assert images.shape[3] == 1
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2]) #images 55,000 , 28*28의 배열로 변경시켜준다.
      if dtype == dtypes.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(numpy.float32) #dtype 이 float32이면 images의 타입을 numpy.float32이로 바꾼다. 
        images = numpy.multiply(images, 1.0 / 255.0) #images값을 255로 나눠서 0~1사이의 값으로 변경해준다.  
    self._images = images  #private데이터로 선언헤주고 아래의 함수를 통해서 값을 가져올수 있게 하였다. 
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)
      ]
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples) # 0~ num_examples-1 의 값을 가지는 배열을 만들어준다.
      numpy.random.shuffle(perm) # 위의 배열을 값을  섞는다.
      self._images = self._images[perm]#perm 의 인덱스대로 _images의 이미지 배열 값을 바꿔준다.
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0 #다시 start 값을 0으로 설정해준다.
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end] #bactchsize 만큼의 랜덤한 image와 라벨을 반환해 준다. 


def read_data_sets(train_dir,
                   fake_data=False,
                   one_hot=False,
                   dtype=dtypes.float32,
                   reshape=True,
                   validation_size=5000): #train_dir -> /tmp/data/ one_hot 을 True로 호출
  if fake_data:

    def fake():
      return DataSet([], [], fake_data=True, one_hot=one_hot, dtype=dtype)

    train = fake()
    validation = fake()
    test = fake()
    return base.Datasets(train=train, validation=validation, test=test)

  TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
  TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
  TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
  TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

  local_file = base.maybe_download(TRAIN_IMAGES, train_dir,
                                   SOURCE_URL + TRAIN_IMAGES) #학습 데이터를 다운로드 하고
  with open(local_file, 'rb') as f:
    train_images = extract_images(f) #이미지 압축을 해제하고 이미지파일들을 객체에 맵핑시켜준다. (갯수,row, col, 1)들어간 4차원 이미지 배열

  local_file = base.maybe_download(TRAIN_LABELS, train_dir,
                                   SOURCE_URL + TRAIN_LABELS)
  with open(local_file, 'rb') as f:
    train_labels = extract_labels(f, one_hot=one_hot)

  local_file = base.maybe_download(TEST_IMAGES, train_dir,
                                   SOURCE_URL + TEST_IMAGES)
  with open(local_file, 'rb') as f:
    test_images = extract_images(f)

  local_file = base.maybe_download(TEST_LABELS, train_dir,
                                   SOURCE_URL + TEST_LABELS)
  with open(local_file, 'rb') as f:
    test_labels = extract_labels(f, one_hot=one_hot)

  if not 0 <= validation_size <= len(train_images):
    raise ValueError(
        'Validation size should be between 0 and {}. Received: {}.'
        .format(len(train_images), validation_size))

  validation_images = train_images[:validation_size] # 0~4999의  이미지를 검증용으로 뽑아낸다. 
  validation_labels = train_labels[:validation_size] # 0~4999의  라벨을 검증용으로  뽑아낸다.
  train_images = train_images[validation_size:] # 5000이후의 이미지를 추출해서 train image 배열로 추출한다.
  train_labels = train_labels[validation_size:] # 5000이후의 라벨을 추출해서 train_labels 배열로 추출한다.

  train = DataSet(train_images, train_labels, dtype=dtype, reshape=reshape)
  validation = DataSet(validation_images,
                       validation_labels,
                       dtype=dtype,
                       reshape=reshape)
  test = DataSet(test_images, test_labels, dtype=dtype, reshape=reshape)

  return base.Datasets(train=train, validation=validation, test=test) #named tuple을 생성해서 구조체 처럼 데이터를 접근해서 쓸 수 있게 해준다. x.train x.validation, x.test


def load_mnist(train_dir='MNIST-data'):
  return read_data_sets(train_dir)
