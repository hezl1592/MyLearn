from sklearn.feature_extraction.text import CountVectorizer


def countvec(text):
    '''文本数据特征值化'''
    # 单个字母不进行统计（单个字母不会影响整个文本的信息）
    print(text)
    textt = CountVectorizer()

    data = textt.fit_transform(text)
    print(textt.get_feature_names())
    print(data)
    print(data.toarray())
    return None


if __name__ == "__main__":
    text = corpus = ['This is the first document.',
                     'This is the second second document.',
                     'And the third one.',
                     'Is this the first document?',
                     ]
    countvec(text)
