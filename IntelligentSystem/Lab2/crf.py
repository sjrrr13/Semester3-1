import numpy as np
import torch


# 从文件中读取特征模板，包括模板对应的id，比如'U00'
def get_template(file: str):
    unigram = []
    bigram = []
    unigram_id = []
    bigram_id = []
    f = open(file, encoding='utf-8')
    # 解析文件的每一行
    for line in f:
        if '#' in line or line.startswith('\n'):
            continue
        temp_id = line[0:line.find(':')]
        temp_list = []
        temp = line[line.find(':') + 1:]
        grams = temp.split('/')
        if len(grams) > 1:
            for gram in grams:
                temp_list.append(int(gram[temp.find('[') + 1: gram.find(',')]))
        else:
            temp_list.append(int(temp[temp.find('[') + 1: temp.find(',')]))

        if line.startswith('U'):
            unigram.append(temp_list)
            unigram_id.append(temp_id)
        else:
            bigram.append(temp_list)
            bigram_id.append(temp_id)
    return unigram, bigram, unigram_id, bigram_id


# 读取训练集
def get_data(file: str):
    sentences = []
    tags = []
    sentence = ''
    tag = ''
    f = open(file, encoding='utf-8')

    for line in f:
        if line != '\n':
            line = line.strip()
            s, t = line.split(' ')
            sentence += s
            tag += t
        else:
            if len(sentence) > 0:
                sentences.append(sentence)
                tags.append(tag)
                sentence = ''
                tag = ''
    return sentences, tags


# 生成特征模板
def get_feature(sentence, position, template_id, template, tag):
    feature = str(template_id)
    for i in template:
        p = position + i
        if p < 0 or p >= len(sentence):
            feature += ' '
        else:
            feature += sentence[p]
    feature += ':' + tag
    return feature


# 根据tag得到对应下标的dictionary
tag_dictionary = {'S': 0, 'B': 1, 'I': 2, 'E': 3}


class CRF:
    def __init__(self):
        self.unigrams, self.bigrams, self.unigram_ids, self.bigram_ids = get_template(
            './data/train_dataset/dataset2/template.utf8')
        self.tags = ['S', 'B', 'I', 'E']
        self.score = {}

    # 初始化特征函数
    def init_feature(self, sentences):
        # print('Start analyzing features')
        for sentence in sentences:
            for i in range(len(sentence)):
                for u in range(len(self.unigrams)):
                    for t in range(4):
                        feature = get_feature(sentence, i, self.unigram_ids[u], self.unigrams[u], self.tags[t])
                        if feature in self.score.keys():
                            continue
                        else:
                            self.score[feature] = 0
                for b in range(len(self.bigrams)):
                    if i == 0:
                        for t1 in range(4):
                            feature = get_feature(sentence, i, self.bigram_ids[b], self.bigrams[b], ' ' + self.tags[t1])
                            if feature in self.score.keys():
                                continue
                            else:
                                self.score[feature] = 0
                    else:
                        for t1 in range(4):
                            for t2 in range(4):
                                feature = get_feature(sentence, i, self.bigram_ids[b], self.bigrams[b],
                                                      self.tags[t1] + self.tags[t2])
                                if feature in self.score.keys():
                                    continue
                                else:
                                    self.score[feature] = 0
        # print('Features successfully analyzed')

    # 根据feature得到unigram的评分
    def get_u_score(self, sentence, position, tag):
        u_score = 0
        for i in range(len(self.unigrams)):
            feature = get_feature(sentence, position, self.unigram_ids[i], self.unigrams[i], tag)
            # if feature in self.score.keys():
            u_score += self.score[feature]
        return u_score

    # 根据feature得到bigram的评分
    def get_b_score(self, sentence, position, prev_tag, tag):
        b_score = 0
        for i in range(len(self.bigrams)):
            feature = get_feature(sentence, position, self.bigram_ids[i], self.bigrams[i], prev_tag + tag)
            # if feature in self.score.keys():
            b_score += self.score[feature]
        return b_score

    # 应用Viterbi算法进行预测
    def predict(self, sentence):
        length = len(sentence)
        max_score = np.zeros((4, length))  # 记录每个tag在每个位置的最高评分
        trans_tag = np.zeros((4, length), dtype=str)  # 记录tag的转移情况
        for i in range(length):
            for j in range(4):
                current_tag = self.tags[j]
                if i == 0:
                    u_score = self.get_u_score(sentence, 0, current_tag)
                    b_score = self.get_b_score(sentence, 0, ' ', current_tag)
                    max_score[j][0] = u_score + b_score
                    trans_tag[j][0] = ''  # 第0个位置的tag的前一个为null
                else:
                    temp_score = []
                    u_score = self.get_u_score(sentence, i, current_tag)
                    for k in range(4):
                        prev_tag = self.tags[k]
                        b_score = self.get_b_score(sentence, i, prev_tag, current_tag)
                        temp_score.append(u_score + b_score + max_score[k][i - 1])
                    max_score[j][i] = np.max(temp_score)
                    trans_tag[j][i] = self.tags[np.argmax(temp_score)]
        # Viterbi算法，逆向查找结果
        result = [''] * length
        last_score = []
        for i in range(4):
            last_score.append(max_score[i][length - 1])
        result[length - 1] = self.tags[np.argmax(last_score)]
        for i in range(length - 2, -1, -1):
            result[i] = trans_tag[tag_dictionary[result[i + 1]]][i + 1]
        # 返回字符串形式的结果
        sequence = ''
        for t in result:
            sequence += t
        return sequence

    # 训练过程
    def train(self, sentence, tag):
        sequence = self.predict(sentence)  # 得到预测结果
        wrong = 0  # 记录错误tag数量
        for i in range(len(sentence)):
            s = sequence[i]
            t = tag[i]
            if s != t:
                wrong += 1
                # 更新unigram的特征函数
                for u in range(len(self.unigrams)):
                    right_u_feature = get_feature(sentence, i, self.unigram_ids[u], self.unigrams[u], t)
                    wrong_u_feature = get_feature(sentence, i, self.unigram_ids[u], self.unigrams[u], s)
                    # 将正确的feature的得分+1，错误的-1
                    self.score[right_u_feature] += 1
                    self.score[wrong_u_feature] -= 1
                # 同上更新bigram的特征函数
                for b in range(len(self.bigrams)):
                    if i == 0:
                        right_b_feature = get_feature(sentence, i, self.bigram_ids[b], self.bigrams[b], ' ' + t)
                        wrong_b_feature = get_feature(sentence, i, self.bigram_ids[b], self.bigrams[b], ' ' + s)
                    else:
                        right_b_feature = get_feature(sentence, i, self.bigram_ids[b], self.bigrams[b],
                                                      tag[i - 1: i + 1])
                        wrong_b_feature = get_feature(sentence, i, self.bigram_ids[b], self.bigrams[b],
                                                      sequence[i - 1: i + 1])
                    self.score[right_b_feature] += 1
                    self.score[wrong_b_feature] -= 1
        return wrong


if __name__ == '__main__':
    epoch = 100  # epoch数量
    s1, t1 = get_data('./data/train_dataset/dataset1/train.utf8')
    s2, t2 = get_data('./data/train_dataset/dataset2/train.utf8')
    sentences = s1 + s2
    tags = t1 + t2
    # sentences, tags = get_data('./data/train_dataset/dataset2/train.utf8')
    model = CRF()
    # model = torch.load('./models/crf/crf2.pth')
    model.init_feature(sentences)
    print('Feature quantity: ', len(model.score))
    save_path = './models/crf/crf5_3.pth'
    history = 0

    for i in range(epoch):
        total = 0
        wrong = 0
        for j in range(len(sentences)):
            sentence = sentences[j]
            tag = tags[j]
            total += len(sentence)
            wrong += model.train(sentence, tag)
        accuracy = (total - wrong) / total * 100
        print('Epoch ' + str(i) + ': Accuracy ' + str(accuracy) + '%' + '; History: ' + str(history) + '%')  # 打印正确率
        if history < accuracy:  # 如果正确率提高的话，保存模型
            history = accuracy
            torch.save(model, save_path)
            print('model saved')
