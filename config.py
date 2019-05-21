import os

class Config(object):
    """配置参数"""
    base_dir = 'data'

    vocab_dir = os.path.join(base_dir, 'vocab.txt')
    save_dir = 'checkpoints/textcnn'
    save_path = os.path.join(save_dir, 'best_validation')

    train_dir = os.path.join(base_dir, 'train.txt')
    test_dir = os.path.join(base_dir, 'test.txt')
    val_dir = os.path.join(base_dir, 'val.txt')
    categories_dir = os.path.join(base_dir, 'categories.txt')

    vocab_size = 5000  # 词汇表大小
    seq_length = 100  # 输入文本的最大长度(输入字符串长度最长的不能超过该长度，如果不够的则用0填充)
    num_classes = 1258  # 分类的类别数

    embedding_dim = 64  # 词向量维度
    num_filters = 256  # 卷积核数目
    kernel_size = 5  # 卷积核尺寸

    hidden_dim = 128  # 全连接层神经元数（一般设为256和num_classes之间的64的倍数）

    dropout_keep_prob = 0.5  # dropout比例,keep_prob= 1 - dropout_rate
    learning_rate = 1e-3  # 学习率

    batch_size = 64  # 每批训练大小
    num_epochs = 3  # 最大的epoch(所有样本训练一次)
    max_epochs = 10  # 最大的epoch(所有样本训练一次)

    print_per_batch = 100  # 每迭代多少次打印loss和准确率
    print_iter = 100  # 每迭代多少次打印loss和准确率
    save_per_batch = 10  # 每迭代多少次存入tensorboard
    save_iter = 10  # 每迭代多少次存入tensorboard




