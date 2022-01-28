import torch
import torch.nn as nn
import torch.optim as optim

# 设置CPU生成随机数的种子，方便下次复现实验结果。
torch.manual_seed(1)


# 返回vec中的最大值的下标
def argmax(vec):
    _, index = torch.max(vec, 1)
    return index.item()


# 将句子中每个字在字典中的值合并，作为tensor返回
def prepare_sequence(seq, to_ix):
    index = [to_ix[w] for w in seq]
    return torch.tensor(index, dtype=torch.long)


# LogSumExp函数，前向算法中用的
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    temp = torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))
    return max_score + temp


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = {"S": 0, "B": 1, "I": 2, 'E': 3, START_TAG: 4, STOP_TAG: 5}
        self.tag_num = len(self.tag_to_ix)
        # 网络结构
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)
        # 将网络的输出转换为tag
        self.hidden2tag = nn.Linear(hidden_dim, self.tag_num)
        # 参数转移矩阵
        self.transitions = nn.Parameter(torch.randn(self.tag_num, self.tag_num))
        # 阻止转移到START_TAG或者从STOP_TAG开始转移
        self.transitions.data[self.tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, self.tag_to_ix[STOP_TAG]] = -10000
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2), torch.randn(2, 1, self.hidden_dim // 2))

    # 前向算法
    def _forward_alg(self, feats):
        init_alphas = torch.full((1, self.tag_num), -10000.)
        # START_TAG分数初始化
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # 实现自动反向算法的变量
        forward_var = init_alphas
        for feat in feats:
            alphas_t = []  # t时刻的alpha的值
            for next_tag in range(self.tag_num):
                # 发射分数与前一步的tag无关
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tag_num)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                # 前向算法当前的tag的分数是所有分数的LogSumExp值
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)

        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    # 解析特征
    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    # 根据tag序列进行评分
    def _score_sentence(self, feats, tags):
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                    self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    # Viterbi解码算法
    def _viterbi(self, feats):
        # 初始化一些变量
        backpointers = []
        init_vvars = torch.full((1, self.tag_num), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var会记录前一次迭代的值
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # 记录当前步的backpointers
            viterbivars_t = []  # 记录当前步的值

            for next_tag in range(self.tag_num):
                # 记录当前步到下一步的转移
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # 加上emission score
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # 转移到STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # 将START_TAG移除
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]
        # 进行倒序
        best_path.reverse()
        return path_score, best_path

    # 负对数似然函数计算loss
    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    # 用Viterbi算法找到评分最高的路径
    def forward(self, sentence):
        lstm_feats = self._get_lstm_features(sentence)
        score, tag_seq = self._viterbi(lstm_feats)
        return score, tag_seq


# 得到训练数据
def get_data(file):
    sentence = []
    tag = []
    data = []
    f = open(file, encoding='utf-8')

    for line in f:
        if line != '\n':
            line = line.strip()
            s, t = line.split(' ')
            sentence.append(s)
            tag.append(t)
        else:
            if len(sentence) > 0:
                data.append((sentence, tag))
                sentence = []
                tag = []
    return data


START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5
HIDDEN_DIM = 4
epoch = 100
if __name__ == '__main__':
    # 将两个训练集的数据拼接到一起
    d1 = get_data('./data/train_dataset/dataset1/train.utf8')
    d2 = get_data('./data/train_dataset/dataset2/train.utf8')
    training_data = d1 + d2

    # 将数据存入字典
    word_to_ix = {}
    for sentence, tags in training_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    # model = BiLSTM_CRF(len(word_to_ix), EMBEDDING_DIM, HIDDEN_DIM)
    model = torch.load('./models/BiLSTM_CRF/Bi3.pth')
    # 用SGD优化器
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    history = 0
    for epoch in range(epoch):
        for sentence, tags in training_data:
            # 清空积攒的梯度
            model.zero_grad()
            # 将输入转换为tensor
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = torch.tensor([model.tag_to_ix[t] for t in tags], dtype=torch.long)
            loss = model.neg_log_likelihood(sentence_in, targets)
            # 计算loss和梯度，并更新参数
            loss.backward()
            optimizer.step()

        # 计算正确率
        with torch.no_grad():
            wrong = 0
            total = 0
            for sentence in training_data:
                words = sentence[0]
                tags = sentence[1]
                total += (len(words))
                precheck_sent = prepare_sequence(words, word_to_ix)
                my_tag = model(precheck_sent)[1]
                for t in range(len(tags)):
                    if model.tag_to_ix[tags[t]] != my_tag[t]:
                        wrong += 1
            accuracy = (total - wrong) / total * 100
            print('Epoch ' + str(epoch) + ': Accuracy: ' + str(accuracy) + '%; History: ' + str(history) + '%')
            if accuracy > history:
                history = accuracy
                torch.save(model, './models/BiLSTM_CRF/Bi3.pth')
                print('model saved')
