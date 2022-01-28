import torch
import torch.nn as nn
import torch.optim as optim

# 相同的几个辅助函数
torch.manual_seed(1)


def argmax(vec):
    index = torch.argmax(vec)
    return index.item()


def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    temp = torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))
    return max_score + temp


# 向量库
class VectorLib:
    def __init__(self, file, encoding='utf8'):
        self.data = dict()
        with open(file, 'r', encoding=encoding) as f:
            line = f.readlines()
            self.vocab_size, self.vector_dim = tuple(map(int, line[0].split()))
            for l in line[1:]:
                arr = l.split()
                word = arr[0]
                vector = torch.tensor(list(map(float, arr[1:])))
                self.data[word] = vector

    def __getitem__(self, key):
        if key in self.data.keys():
            return self.data[key]
        else:  # 若查找的key不在vocabulary里则随机赋值
            self.data[key] = torch.randn(self.vector_dim)
            return self.data[key]

    def __len__(self):
        return len(self.data)


# 根据文件得到数据集
class DataSet:
    def __init__(self, file, vector_lib, encoding='utf8'):
        self.tag_to_ix = {}
        self.data = []
        self.vector_lib = vector_lib

        with open(file, 'r', encoding=encoding) as f:
            line = f.readlines()
            sentence = ([], [])
            for l in line:
                i = l.strip()
                if len(i) > 0:
                    tag = 'NIL'
                    i = i.split()
                    if len(i) == 1:
                        sentence[0].append(i[0])
                        sentence[1].append(tag)
                    else:
                        word, tag = i
                        sentence[0].append(word)
                        sentence[1].append(tag)
                    if tag not in self.tag_to_ix:
                        self.tag_to_ix[tag] = len(self.tag_to_ix)
                else:
                    self.data.append(sentence)
                    sentence = ([], [])

            if len(sentence) > 0:
                self.data.append(sentence)

    # 返回word对应的vector
    def prepare_word_seq(self, seq):
        return [self.vector_lib[w] for w in seq]

    # 返回tag对应的值
    def prepare_tag_seq(self, seq):
        index = [self.tag_to_ix[w] for w in seq]
        return torch.tensor(index, dtype=torch.long)

    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return len(self.data)


# 与另一个BiLSTM_CRF类区别不大，不做注释
class BiLSTM_CRF(nn.Module):
    def __init__(self, input_dim, hidden_dim, tag_to_ix):
        super(BiLSTM_CRF, self).__init__()
        self.hidden_dim = hidden_dim
        self.tag_to_ix = tag_to_ix
        # 增加START_TAG和STOP_TAG，在tag_to_ix中添加相应的项
        self.START_TAG = '<START_TAG>'
        self.STOP_TAG = '<STOP_TAG>'
        self.tag_to_ix[self.START_TAG] = len(self.tag_to_ix)
        self.tag_to_ix[self.STOP_TAG] = len(self.tag_to_ix)
        self.START_TAG = self.tag_to_ix[self.START_TAG]
        self.STOP_TAG = self.tag_to_ix[self.STOP_TAG]
        self.tag_num = len(self.tag_to_ix)
        # 网络结构
        self.lstm = nn.LSTM(input_dim, hidden_dim // 2, num_layers=1, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim, self.tag_num)
        self.transitions = nn.Parameter(torch.randn(self.tag_num, self.tag_num))
        self.transitions.data[self.START_TAG, :] = -10000
        self.transitions.data[:, self.STOP_TAG] = -10000
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2, ), torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        init_alphas = torch.full((1, self.tag_num), -10000.)
        init_alphas[0][self.START_TAG] = 0.

        forward_vars = init_alphas
        for feat in feats:
            alphas_t = []
            for next_tag in range(self.tag_num):
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tag_num)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_vars + trans_score + emit_score
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_vars = torch.cat(alphas_t).view(1, -1)

        terminal_var = forward_vars + self.transitions[self.STOP_TAG]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sequence):
        size = len(sequence)
        self.hidden = self.init_hidden()
        embeds = torch.cat(sequence).view(size, 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(size, self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.START_TAG], dtype=torch.long), tags])
        for (i, feat) in enumerate(feats):
            emit_score = feat[tags[i + 1]]
            trans_score = self.transitions[tags[i + 1], tags[i]]
            score = score + emit_score + trans_score
        score = score + self.transitions[self.STOP_TAG, tags[-1]]
        return score

    def _viterbi(self, feats):
        backpointers = []
        init_vvars = torch.full((1, self.tag_num), -10000.)
        init_vvars[0][self.START_TAG] = 0

        forward_vars = init_vvars
        for feat in feats:
            bptrs = []
            viterbivars = []

            for next_tag in range(self.tag_num):
                next_tag_vars = forward_vars + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_vars)
                bptrs.append(best_tag_id)
                viterbivars.append(next_tag_vars[0][best_tag_id].view(1))
            forward_vars = (torch.cat(viterbivars) + feat).view(1, -1)
            backpointers.append(bptrs)

        terminal_var = forward_vars + self.transitions[self.STOP_TAG]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]
        best_path = [best_tag_id]
        for bptrs in reversed(backpointers):
            best_tag_id = bptrs[best_tag_id]
            best_path.append(best_tag_id)

        start = best_path.pop()
        assert start == self.START_TAG
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):
        lstm_feats = self._get_lstm_features(sentence)
        score, best_seq = self._viterbi(lstm_feats)
        return score, best_seq


def predict(sentence, model, vector_lib):
    sentence_in = [vector_lib[w] for w in sentence]
    _, label = model(sentence_in)
    result = ""
    for i in label:
        if i == 0:
            result += 'S'
        elif i == 1:
            result += 'B'
        elif i == 2:
            result += 'E'
        elif i == 3:
            result += 'I'
        else:
            continue
    return result


if __name__ == '__main__':
    epoch = 100
    vector_lib = VectorLib('./vector_library.utf8')
    data_set = DataSet('./data/train_dataset/dataset1/train.utf8', vector_lib)
    # model = BiLSTM_CRF(vector_lib.vector_dim, 150, data_set.tag_to_ix)
    model = torch.load('./models/BiLSTM_CRF/PreBi1.pth')
    # 用SGD优化器
    optimizer = optim.SGD(model.parameters(), lr=0.005, weight_decay=1e-4)
    for e in range(epoch):
        for (i, (sentence, tags)) in enumerate(data_set):
            model.zero_grad()
            sentence_in = data_set.prepare_word_seq(sentence)
            tag_set = data_set.prepare_tag_seq(tags)
            loss = model.neg_log_likelihood(sentence_in, tag_set)
            loss.backward()
            optimizer.step()
        print('Epoch ' + str(e) + ': Loss: ' + str(loss.item()))
        torch.save(model, './models/BiLSTM_CRF/PreBi1.pth')
    # print(predict('丰子恺当年送他三句话：“多读书，广结交，少说话', model, vector_lib))
