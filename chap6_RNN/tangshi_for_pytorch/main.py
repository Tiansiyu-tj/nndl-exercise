import numpy as np
import collections
import torch
from torch.autograd import Variable
import torch.optim as optim

import rnn as rnn_lstm

start_token = 'G'
end_token = 'E'
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 生成阶段缓存，避免重复加载词表和模型
_gen_cache = None


def process_poems1(file_name):
    """

    :param file_name:
    :return: poems_vector  have tow dimmention ,first is the poem, the second is the word_index
    e.g. [[1,2,3,4,5,6,7,8,9,10],[9,6,3,8,5,2,7,4,1]]

    """
    poems = []
    with open(file_name, "r", encoding='utf-8', ) as f:
        for line in f.readlines():
            try:
                # 只按第一个冒号切分，避免内容中出现冒号导致解析失败
                title, content = line.strip().split(':', 1)
                # content = content.replace(' ', '').replace('，','').replace('。','')
                content = content.replace(' ', '')
                if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content or \
                                start_token in content or end_token in content:
                    continue
                if len(content) < 5 or len(content) > 80:
                    continue
                content = start_token + content + end_token
                poems.append(content)
            except ValueError as e:
                pass
    # 按诗的字数排序
    poems = sorted(poems, key=lambda line: len(line))
    # print(poems)
    # 统计每个字出现次数
    all_words = []
    for poem in poems:
        all_words += [word for word in poem]
    counter = collections.Counter(all_words)  # 统计词和词频。
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])  # 排序
    words, _ = zip(*count_pairs)
    words = words[:len(words)] + (' ',)
    word_int_map = dict(zip(words, range(len(words))))
    poems_vector = [list(map(word_int_map.get, poem)) for poem in poems]
    return poems_vector, word_int_map, words

def process_poems2(file_name):
    """
    :param file_name:
    :return: poems_vector  have tow dimmention ,first is the poem, the second is the word_index
    e.g. [[1,2,3,4,5,6,7,8,9,10],[9,6,3,8,5,2,7,4,1]]

    """
    poems = []
    with open(file_name, "r", encoding='utf-8', ) as f:
        # content = ''
        for line in f.readlines():
            try:
                line = line.strip()
                if line:
                    content = line.replace(' '' ', '').replace('，','').replace('。','')
                    if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content or \
                                    start_token in content or end_token in content:
                        continue
                    if len(content) < 5 or len(content) > 80:
                        continue
                    # print(content)
                    content = start_token + content + end_token
                    poems.append(content)
                    # content = ''
            except ValueError as e:
                # print("error")
                pass
    # 按诗的字数排序
    poems = sorted(poems, key=lambda line: len(line))
    # print(poems)
    # 统计每个字出现次数
    all_words = []
    for poem in poems:
        all_words += [word for word in poem]
    counter = collections.Counter(all_words)  # 统计词和词频。
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])  # 排序
    words, _ = zip(*count_pairs)
    words = words[:len(words)] + (' ',)
    word_int_map = dict(zip(words, range(len(words))))
    poems_vector = [list(map(word_int_map.get, poem)) for poem in poems]
    return poems_vector, word_int_map, words

def generate_batch(batch_size, poems_vec, word_to_int):
    n_chunk = len(poems_vec) // batch_size
    x_batches = []
    y_batches = []
    for i in range(n_chunk):
        start_index = i * batch_size
        end_index = start_index + batch_size
        x_data = poems_vec[start_index:end_index]
        y_data = []
        for row in x_data:
            y  = row[1:]
            y.append(row[-1])
            y_data.append(y)
        """
        x_data             y_data
        [6,2,4,6,9]       [2,4,6,9,9]
        [1,4,2,8,5]       [4,2,8,5,5]
        """
        # print(x_data[0])
        # print(y_data[0])
        # exit(0)
        x_batches.append(x_data)
        y_batches.append(y_data)
    return x_batches, y_batches


def run_training():
    # 处理数据集
    # poems_vector, word_to_int, vocabularies = process_poems2('./tangshi.txt')
    poems_vector, word_to_int, vocabularies = process_poems1('./poems.txt')
    # 生成batch
    print("finish  loadding data")
    print("using device:", device)
    BATCH_SIZE = 800#修改原版代码成GPU版本，同时batch 100-->600(4G显存占用)

    torch.manual_seed(5)
    word_embedding = rnn_lstm.word_embedding( vocab_length= len(word_to_int) + 1 , embedding_dim= 100)
    rnn_model = rnn_lstm.RNN_model(batch_sz = BATCH_SIZE,vocab_len = len(word_to_int) + 1 ,word_embedding = word_embedding ,embedding_dim= 100, lstm_hidden_dim=128)
    rnn_model = rnn_model.to(device)

    # optimizer = optim.Adam(rnn_model.parameters(), lr= 0.001)
    optimizer=optim.RMSprop(rnn_model.parameters(), lr=0.01)

    loss_fun = torch.nn.NLLLoss()
    # rnn_model.load_state_dict(torch.load('./poem_generator_rnn'))  # if you have already trained your model you can load it by this line.

    for epoch in range(30):
        batches_inputs, batches_outputs = generate_batch(BATCH_SIZE, poems_vector, word_to_int)
        n_chunk = len(batches_inputs)
        for batch in range(n_chunk):
            batch_x = batches_inputs[batch]
            batch_y = batches_outputs[batch] # (batch , time_step)

            loss = 0
            for index in range(BATCH_SIZE):
                x = np.array(batch_x[index], dtype = np.int64)
                y = np.array(batch_y[index], dtype = np.int64)
                x = Variable(torch.from_numpy(np.expand_dims(x,axis=1))).to(device)
                y = Variable(torch.from_numpy(y )).to(device)
                pre = rnn_model(x)
                loss += loss_fun(pre , y)
                if index == 0:
                    _, pre = torch.max(pre, dim=1)
                    print('prediction', pre.detach().cpu().tolist()) # the following  three line can print the output and the prediction
                    print('b_y       ', y.detach().cpu().tolist())   # And you need to take a screenshot and then past is to your homework paper.
                    print('*' * 30)
            loss  = loss  / BATCH_SIZE
            print("epoch  ",epoch,'batch number',batch,"loss is: ", loss.detach().cpu().item())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(rnn_model.parameters(), 1)
            optimizer.step()

            if batch % 20 ==0:
                torch.save(rnn_model.state_dict(), './poem_generator_rnn')
                print("finish  save model")



def to_word(predict, vocabs):  # 预测的结果转化成汉字
    sample = np.argmax(predict)

    if sample >= len(vocabs):
        sample = len(vocabs) - 1

    return vocabs[sample]


def pretty_print_poem(poem):  # 令打印的结果更工整
    # 去掉起止标记后再按句号切分，避免生成结果被误过滤
    clean_poem = poem.replace(start_token, '').split(end_token)[0]
    poem_sentences = [s for s in clean_poem.split('。') if s]
    if poem_sentences:
        for s in poem_sentences:
            print(s + '。')
    else:
        print(clean_poem)


def _load_generation_resources():
    """加载并缓存生成阶段需要的词表与模型。"""
    global _gen_cache
    if _gen_cache is not None:
        return _gen_cache

    poems_vector, word_int_map, vocabularies = process_poems1('./poems.txt')
    word_embedding = rnn_lstm.word_embedding(vocab_length=len(word_int_map) + 1, embedding_dim=100)
    rnn_model = rnn_lstm.RNN_model(batch_sz=64, vocab_len=len(word_int_map) + 1, word_embedding=word_embedding,
                                   embedding_dim=100, lstm_hidden_dim=128)
    rnn_model = rnn_model.to(device)
    rnn_model.load_state_dict(torch.load('./poem_generator_rnn', map_location=device))
    rnn_model.eval()

    _gen_cache = (word_int_map, vocabularies, rnn_model)
    return _gen_cache


def _pick_next_word(log_prob_row, vocabularies, current_len, min_len=12):
    """选择下一个字，避免过早输出结束标记导致内容过短。"""
    sorted_ids = np.argsort(np.array(log_prob_row))[::-1]
    for idx in sorted_ids:
        next_word = vocabularies[idx] if idx < len(vocabularies) else ' '
        if next_word == end_token and current_len < min_len:
            continue
        return next_word
    return end_token


def gen_poem(begin_word):
    # poems_vector, word_int_map, vocabularies = process_poems2('./tangshi.txt')  #  use the other dataset to train the network
    word_int_map, vocabularies, rnn_model = _load_generation_resources()

    if begin_word not in word_int_map:
        return begin_word + '（起始字不在词表中）'

    # 训练时样本都以 start_token 开头，生成时保持一致
    poem = start_token + begin_word
    word = begin_word
    while word != end_token:
        input = np.array([word_int_map.get(w, word_int_map[' ']) for w in poem], dtype=np.int64)
        input = Variable(torch.from_numpy(input)).to(device)
        output = rnn_model(input, is_test=True)
        word = _pick_next_word(output.detach().cpu().tolist()[-1], vocabularies, current_len=len(poem))
        poem += word
        # print(word)
        # print(poem)
        if len(poem) > 40:
            break
    return poem



#run_training()  # 如果不是训练阶段 ，请注销这一行 。 网络训练时间很长。


pretty_print_poem(gen_poem("日"))
print("\n")
pretty_print_poem(gen_poem("红"))
print("\n")
pretty_print_poem(gen_poem("山"))
print("\n")
pretty_print_poem(gen_poem("夜"))
print("\n")
pretty_print_poem(gen_poem("湖"))
print("\n")
pretty_print_poem(gen_poem("湖"))
print("\n")
pretty_print_poem(gen_poem("湖"))
print("\n")
pretty_print_poem(gen_poem("君"))


