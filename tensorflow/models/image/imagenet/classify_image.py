# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Simple image classification with Inception.

Run image classification with Inception trained on ImageNet 2012 Challenge data
set.

This program creates a graph from a saved GraphDef protocol buffer,
and runs inference on an input JPEG image. It outputs human readable
strings of the top 5 predictions along with their probabilities.

Change the --image_file argument to any jpg image to compute a
classification of that image.

Please see the tutorial and website for a detailed description of how
to use this script to perform image recognition.

https://tensorflow.org/tutorials/image_recognition/
"""

from __future__ import absolute_import #상대주소를 절대주소로 바꾸는 모듈 
from __future__ import division #나눗셈 연산자를 바꾼다 Inteager를 Float으로 자동으로 바꿔주는 것
from __future__ import print_function #프린트 함수

import argparse
import os.path #파일의 경로를 수정 생성하는 것을 쉽게 다룰 수 있도록 하는 함수 
import re #정규 표현식 
import sys #system에 관련된 함수 인자를 받을 때 쓰는 함수
import tarfile #압축해제 또는 압축하는 모듈

import numpy as np #숫자 관련된 함수를 C라이르버리로 최적화한 모듈
from six.moves import urllib #url을 불러올 때 쓰는 모듈
import tensorflow as tf #tensorflow 모듈

FLAGS = None

# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz' # URL설정
# pylint: enable=line-too-long


class NodeLookup(object):
  """Converts integer node ID's to human readable labels."""

  def __init__(self,
               label_lookup_path=None,
               uid_lookup_path=None):
    if not label_lookup_path:
      label_lookup_path = os.path.join(
          FLAGS.model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
    if not uid_lookup_path:
      uid_lookup_path = os.path.join(
          FLAGS.model_dir, 'imagenet_synset_to_human_label_map.txt')
    self.node_lookup = self.load(label_lookup_path, uid_lookup_path) #node_lookup에 load로 만들어 놓은 node_id_to_name 의 key value 의 Dictionary로 저장 된다.

  def load(self, label_lookup_path, uid_lookup_path):
    """Loads a human readable English name for each softmax node.

    Args:
      label_lookup_path: string UID to integer node ID.
      uid_lookup_path: string UID to human-readable string.

    Returns:
      dict from integer node ID to human-readable string.
    """
    if not tf.gfile.Exists(uid_lookup_path):
      tf.logging.fatal('File does not exist %s', uid_lookup_path)
    if not tf.gfile.Exists(label_lookup_path):
      tf.logging.fatal('File does not exist %s', label_lookup_path)

    # Loads mapping from string UID to human-readable string
    proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
    uid_to_human = {}
    p = re.compile(r'[n\d]*[ \S,]*') #(n은 n 문자 \d는 숫자) *로 이 둘중 하나가 들어있는 여러개를 인식 \S,여러개 ...
                                     # n000004475 organism, being 이런 형태로 연결된 문자열이 있는 식을 배열 형태로 만들어준다.
    for line in proto_as_ascii_lines:
      parsed_items = p.findall(line)
      uid = parsed_items[0]
      human_string = parsed_items[2]
      uid_to_human[uid] = human_string

    # Loads mapping from string UID to integer node ID.
    node_id_to_uid = {}
    proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
      if line.startswith('  target_class:'):
        target_class = int(line.split(': ')[1])
      if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1]
        node_id_to_uid[target_class] = target_class_string[1:-2]
        #entry {
        #target_class : 449
        #target_class_string : "n01440764"
        #}  ->여기에서 taget_class와 node_id_to_uid 값을 매핑해준다

    # Loads the final mapping of integer node ID to human-readable string
    node_id_to_name = {}
    for key, val in node_id_to_uid.items():
      if val not in uid_to_human:
        tf.logging.fatal('Failed to locate: %s', val)
      name = uid_to_human[val]
      node_id_to_name[key] = name
      #n000004475 organism, being의 값을 받아서 이 값들을  node_id_to_name 에 key와 이름으로 값을 넣어준다.
    return node_id_to_name

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]


def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(os.path.join(
      FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f: #with as 는 구문인데 , 뭔가를 처리 할 때 여러줄을 처리를 f하나로 처리 한다. 파일을 열고 닫는 것을 자동으로 해준다.
    graph_def = tf.GraphDef() #그래프의 기본 포맷을 가지고 있는 default, prtobuff 파일을 텍스트 형식으로 변경을 해주는 //
    graph_def.ParseFromString(f.read()) #스트링 파일로부터 그래프로 파싱하는 함수
    _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image(image):
  """Runs inference on an image.

  Args:
    image: Image file name.

  Returns:
    Nothing
  """
  if not tf.gfile.Exists(image): #파일이 있는지 확인하고  image파일을 파일을 메모리에 image_data객체로 읽는다.
    tf.logging.fatal('File does not exist %s', image)
  image_data = tf.gfile.FastGFile(image, 'rb').read()

  # Creates graph from saved GraphDef.
  create_graph() #그래프를 생성한다.

  with tf.Session() as sess:
    # Some useful tensors:
    # 'softmax:0': A tensor containing the normalized prediction across
    #   1000 labels.
    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
    #   float description of the image.
    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
    #   encoding of the image.
    # Runs the softmax tensor by feeding the image_data as input to the graph.
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0') #softmax layter  를 거친 tensor 를 포함
    predictions = sess.run(softmax_tensor,
                           {'DecodeJpeg/contents:0': image_data}) #image_data 를 decode하여 bmp형태로 바꿔준다.
    predictions = np.squeeze(predictions) #데이터의 구조를 1차원으로 변경해준다.

    # Creates node ID --> English string lookup.
    node_lookup = NodeLookup()

    top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1] #역순으로 정렬한다. num_top_predictions: 5라는 값이 들어있다. argsort는 index들을 sort해서 준다. 그래서 가장 큰 값을 가지고 있는 class 값을 찾아와준다.
    for node_id in top_k:
      human_string = node_lookup.id_to_string(node_id)
      score = predictions[node_id] #predictions에서 class의 index값에 맞는 score를 찾는다.
      print('%s (score = %.5f)' % (human_string, score))


def maybe_download_and_extract():
  """Download and extract model tar file."""
  dest_directory = FLAGS.model_dir #path /tmp/imagenet
  if not os.path.exists(dest_directory): # 경로가 있는지 확인하고 없으면 directory를 만든다
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1] # /로 구분자를 잡고 -1로 하여 뒤에서부터 순서를 잡아서 배열순서로 찾아준다.
  filepath = os.path.join(dest_directory, filename) #스트링연결 하는데 맨끝에 /를 붙여서 스트링을 연결해준다. dest_directory/filename 으로 만들어준다.
  if not os.path.exists(filepath): #file path가 있는지 확인하고 없으면 다운로드받는다.
    def _progress(count, block_size, total_size): #내장함수 현재 진행상태를 출력해주는 함수
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress) #,_여러개의 값을 리턴받을 때 사용한다. _는 사용하지 않을 값일 경우 빈 객체로 만들어준다.  urlretrieve 파일을 받으면서 _progress함수를 실행시켜준다.
    print() # 한줄뛰어쓰는 \r\n  개행 newline
    statinfo = os.stat(filepath) #다운받은 파일의 정보를 읽어 staeinfo에 값을 저장
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.') #파일의 정보를 출력 파일의 크기 값을 출겨해준다. print에 , 를 넣어주면 연속으로 출력을 해준다. 
  tarfile.open(filepath, 'r:gz').extractall(dest_directory) # 압축해제 readonly gz파일 압축 형태 


def main(_): #def 함수 선언 def 함수명()  들여쓰기를 통해서 괄호를 대신함
  maybe_download_and_extract()
  image = (FLAGS.image_file if FLAGS.image_file else
           os.path.join(FLAGS.model_dir, 'cropped_panda.jpg')) #argument로 --image_file 을 설정했을 경우 그 파일을 FLAGS.image_file의 플래그로 사용하고 정의되어 있지 않을 경우 cropped_panda.jpg를 사용한다.
  run_inference_on_image(image)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # classify_image_graph_def.pb:
  #   Binary representation of the GraphDef protocol buffer.
  # imagenet_synset_to_human_label_map.txt:
  #   Map from synset ID to a human readable string.
  # imagenet_2012_challenge_label_map_proto.pbtxt:
  #   Text representation of a protocol buffer mapping a label to synset ID.
  parser.add_argument(
      '--model_dir',
      type=str,
      default='/tmp/imagenet',
      help="""\
      Path to classify_image_graph_def.pb,
      imagenet_synset_to_human_label_map.txt, and
      imagenet_2012_challenge_label_map_proto.pbtxt.\
      """
  )
  parser.add_argument(
      '--image_file',
      type=str,
      default='',
      help='Absolute path to image file.'
  )
  parser.add_argument(
      '--num_top_predictions',
      type=int,
      default=5,
      help='Display this many predictions.'
  )
  FLAGS = parser.parse_args()

  tf.app.run()
