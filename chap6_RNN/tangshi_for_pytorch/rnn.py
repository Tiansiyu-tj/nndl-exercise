import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np


def weights_init(m):
    """
    只对 Linear 层做 Xavier uniform 初始化。
    """
    classname = m.__class__.__name__  # 当前模块的类名
    if classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
        print("inital  linear weight ")


class word_embedding(nn.Module):
    def __init__(self, vocab_length, embedding_dim):
        super(word_embedding, self).__init__()
        # 词向量矩阵形状为 [词表大小, 词向量维度]
        w_embeding_random_intial = np.random.uniform(
            -1, 1, size=(vocab_length, embedding_dim)
        )
        self.word_embedding = nn.Embedding(vocab_length, embedding_dim)
        self.word_embedding.weight.data.copy_(torch.from_numpy(w_embeding_random_intial))

    def forward(self, input_sentence):
        """
        :param input_sentence: 若干字对应的索引张量
        :return: 对应的词向量序列
        """
        # 根据索引查表，得到每个字的 embedding
        sen_embed = self.word_embedding(input_sentence)
        return sen_embed


class RNN_model(nn.Module):
    def __init__(self, batch_sz, vocab_len, word_embedding, embedding_dim, lstm_hidden_dim):
        super(RNN_model, self).__init__()

        self.word_embedding_lookup = word_embedding
        self.batch_size = batch_sz
        self.vocab_length = vocab_len
        self.word_embedding_dim = embedding_dim
        self.lstm_dim = lstm_hidden_dim

        # 两层 LSTM。
        # input_size: 每个时间步输入一个 embedding_dim 维词向量
        # hidden_size: 每个时间步输出一个 lstm_hidden_dim 维隐藏状态
        # batch_first=True: 输入输出格式均为 (batch, seq, feature)
        self.rnn_lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=2,
            batch_first=True
        )

        # 将每个时间步的隐藏状态映射到整个词表，得到“下一个字”的打分
        self.fc = nn.Linear(lstm_hidden_dim, vocab_len)
        self.apply(weights_init)

        # 输出对数概率，便于后续与 NLLLoss 搭配
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sentence, is_test=False):
        # sentence 中保存的是字的索引，先查 embedding
        # view(1, -1, embedding_dim) 后，张量形状为 [batch=1, seq_len, embedding_dim]
        batch_input = self.word_embedding_lookup(sentence).view(1, -1, self.word_embedding_dim)

        # 两层 LSTM 的初始隐藏状态 h0 和记忆状态 c0 都置零
        h0 = torch.zeros(2, batch_input.size(0), self.lstm_dim, device=batch_input.device)
        c0 = torch.zeros(2, batch_input.size(0), self.lstm_dim, device=batch_input.device)

        # output 形状为 [batch, seq_len, hidden_size]
        output, _ = self.rnn_lstm(batch_input, (h0, c0))

        # 展平 batch 和时间维，得到 [seq_len, hidden_size]
        out = output.contiguous().view(-1, self.lstm_dim)

        # 映射到词表大小，得到每个位置对所有字的打分
        out = F.relu(self.fc(out))

        # 转成对数概率
        out = self.softmax(out)

        if is_test:
            # 生成时只关心最后一个时间步的输出
            prediction = out[-1, :].view(1, -1)
            output = prediction
        else:
            # 训练时保留整个序列每个位置的输出
            output = out
        return output
