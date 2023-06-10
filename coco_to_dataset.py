import json
import re
import pandas as pd
from collections import Counter


SOURCE_ROOT = './dataset'
OUTPUT_ROOT = './dataset/coco_processed_eng'

train_labels_filepath = f'{SOURCE_ROOT}/vqa_train.json'
train_dataset_filepath = f'{OUTPUT_ROOT}/train.csv'
val_labels_filepath = f'{SOURCE_ROOT}/vqa_val.json'
val_dataset_filepath = f'{OUTPUT_ROOT}/val.csv'
test_labels_filepath = f'{SOURCE_ROOT}/vqa_test.json'
test_dataset_filepath = f'{OUTPUT_ROOT}/test.csv'

answer_space_filepath = f'{OUTPUT_ROOT}/answer_space.txt'

TRAIN_SIZE = 400_000
VAL_SIZE = 100_000
TEST_SIZE = 10_000
MAX_WORDS = 1
VOCAB_SIZE = 2000
TRANSLATE_TO_UKR = False

# train
train_data = pd.DataFrame(columns=['question', 'answer', 'image'], index=range(TRAIN_SIZE))

with open(train_labels_filepath, 'r') as file:
    labels = json.load(file)

i = 0
for label in labels[:TRAIN_SIZE]:
    question = label['question'] # maybe lowercase
    answer = label['answer'][0]

    # answer = re.split('\s+', answer)[:max_words]
    answer = re.split('\s+', answer)

    if len(answer) <= MAX_WORDS:
        answer = ' '.join(answer)
        image = label['image']

        train_data.iloc[i, :] = [question, answer, image]
        i += 1

train_data.dropna(how='all', inplace=True)
train_data.to_csv(train_dataset_filepath, index=False)

# val
val_data = pd.DataFrame(columns=['question', 'answer', 'image'], index=range(VAL_SIZE))

with open(val_labels_filepath, 'r') as file:
    labels = json.load(file)

i = 0
for label in labels[:VAL_SIZE]:
    question = label['question']
    answer = label['answer'][0]

    # answer = re.split('\s+', answer)[:max_words]
    answer = re.split('\s+', answer)

    if len(answer) <= MAX_WORDS:
        answer = ' '.join(answer)
        image = label['image']

        val_data.iloc[i, :] = [question, answer, image]
        i += 1

val_data.dropna(how='all', inplace=True)
val_data.to_csv(val_dataset_filepath, index=False)

# test
test_data = pd.DataFrame(columns=['question', 'image'], index=range(TEST_SIZE))

with open(test_labels_filepath, 'r') as file:
    labels = json.load(file)

for i, label in enumerate(labels[:TEST_SIZE]):
    question = label['question']
    image = label['image']

    test_data.iloc[i, :] = [question, image]

test_data.to_csv(test_dataset_filepath, index=False)

# vocabulary
train_counts = Counter(train_data['answer'])
val_counts = Counter(val_data['answer'])

total_counts = train_counts + val_counts

with open(answer_space_filepath, 'w') as file:
    for word, count in total_counts.most_common(VOCAB_SIZE):
        file.write(f'{word}\n')
