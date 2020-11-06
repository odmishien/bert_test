import csv
import sys
from sklearn.model_selection import train_test_split

maxInt = sys.maxsize
csv.field_size_limit(maxInt)

keywords = [
    '結婚',
    '離婚',
    '不倫',
    'デート',
    '恋愛',
    '愛',
    '交際',
    'セックス',
    '出会う',
    '彼氏',
    '彼女',
    '恋人']

with open('data/for_bert/hot_entry.tsv', mode='r', newline='', encoding='utf-8') as f:
    tsv_reader = csv.reader(f, delimiter='\t')
    hot_read_data = []
    for row in tsv_reader:
        for keyword in keywords:
            if keyword in row[0]:
                hot_read_data.append(row)
                break

hot_train_data, hot_test_data = train_test_split(
    hot_read_data, test_size=0.2)

with open('data/for_bert/entry.tsv', mode='r', newline='', encoding='utf-8') as f:
    tsv_reader = csv.reader(f, delimiter='\t')
    read_data = []
    for row in tsv_reader:
        for keyword in keywords:
            if keyword in row[0]:
                read_data.append(row)
                break
    read_data = read_data[:len(hot_read_data)]

train_data, test_data = train_test_split(read_data, test_size=0.2)

train_data = hot_train_data + train_data
test_data = hot_test_data + test_data

header = ["text", "label"]

with open('data/for_bert/train_love.tsv', mode='w', newline='', encoding='utf-8') as f:
    tsv_writer = csv.writer(f, delimiter='\t')
    tsv_writer.writerow(header)
    tsv_writer.writerows(train_data)

with open('data/for_bert/test_love.tsv', mode='w', newline='', encoding='utf-8') as f:
    tsv_writer = csv.writer(f, delimiter='\t')
    tsv_writer.writerow(header)
    tsv_writer.writerows(test_data)
