# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import tensorflow as tf
import tensorflow.keras as kr

from config import Config
from model import TextCNN
from file_reader import read_vocab, read_category, build_vocab

import argparse


# 从键盘读入输入序列，输出预测的类别
def test_file(path):
    contents = []
    from file_reader import native_content
    with open(path, 'r', encoding='utf-8') as f:
        list1 = f.readlines()
        for i in list1:
            sentence = i.rstrip('\n')
            contents.append(native_content(sentence))

    data_id = []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_test = kr.preprocessing.sequence.pad_sequences(data_id, config.seq_length)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=session, save_path=config.save_path)  # 读取保存的模型

        feed_dict = {
            model.input_x: x_test,
            model.keep_prob: 1.0
        }

        y_pred_cls = session.run(model.y_pred_cls, feed_dict=feed_dict)

        print('\033[1;33m')
        file_lines = len(y_pred_cls)
        for i in range(file_lines):
            print('%d/%d:%s' % (i+1, file_lines, categories[y_pred_cls[i]]))

        '''
        y_pred_cls = y_pred_cls[0]

        re = dict(zip(y_pred_cls, range(config.num_classes)))
        # 按概率排序
        ks = sorted(re.keys(), reverse=True)
        categories,_ = read_category()


        # 输出概率最高的三个的概率
        for k in range(3):
            print(categories[re[ks[k]]] + ': %.2f%%' % (ks[k] * 100.0))
        '''


# 从键盘读入输入序列，输出预测的类别
def test_input():
    contents = []
    # 輸入
    sentence = input('\033[1;33minput:')
    print('\033[0m')

    from file_reader import native_content
    contents.append(native_content(sentence))

    data_id = []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_test = kr.preprocessing.sequence.pad_sequences(data_id, config.seq_length)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=session, save_path=config.save_path)  # 读取保存的模型

        feed_dict = {
            model.input_x: x_test,
            model.keep_prob: 1.0
        }

        y_pred_cls = session.run(model.y_, feed_dict=feed_dict)
        y_pred_cls = y_pred_cls[0]

        re = dict(zip(y_pred_cls, range(config.num_classes)))
        # 按概率排序
        ks = sorted(re.keys(), reverse=True)
        categories,_ = read_category()

        print('\033[1;32m')
        # 输出概率最高的三个的概率
        for k in range(3):
            print(categories[re[ks[k]]] + ': %.2f%%' % (ks[k] * 100.0))


def parse_args():
    parser = argparse.ArgumentParser(description='run python test.py -i or python test.py -p testpath.txt')

    parser.add_argument('--path', '-p', type=str, help='path to test dataset')

    parser.add_argument('--input', '-i', action='store_true', help='from keyboard input')

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()

    if not args.path and not args.input:
        print("Args error")
        print("""usage: python test.py [-i / -p test.txt]""")
        exit(0)

    print('Restore CNN model,Please wait...')
    config = Config()
    categories, cat_to_id = read_category()
    words, word_to_id = read_vocab(config.vocab_dir)
    config.vocab_size = len(words)
    model = TextCNN(config)

    if args.path:
        path = args.path
        if not os.path.isfile(path):
            print("IO error")
            print("Read '%s' failed, file does not exist." % path)
            exit(0)
        test_file(path)
    elif args.input:
        test_input()
