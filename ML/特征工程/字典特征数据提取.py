from sklearn.feature_extraction import DictVectorizer


def dicvec(example):
    '''字典数据提取'''
    print(example)
    # dict1 = DictVectorizer()
    dict1 = DictVectorizer(sparse=False)

    data = dict1.fit_transform(example)
    print(dict1.get_feature_names())
    print(data)
    return None


if __name__ == "__main__":
    example = [{'city': '北京', 'temperature': 100},
               {'city': '上海', 'temperature': 60}, 
               {'city': '深圳', 'temperature': 30}]
    dicvec(example)
