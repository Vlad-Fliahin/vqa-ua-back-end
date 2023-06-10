import pandas as pd
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer
import torch
from translate import Translator

translator = Translator(to_lang="uk")

# Load the English-to-Ukrainian translation model and tokenizer
MODEL_NAME = "Helsinki-NLP/opus-mt-en-uk"
SOURCE_ROOT = './dataset/coco_processed_eng'
OUTPUT_ROOT = './dataset/coco_processed_ukr'
MODEL_BATCH_SIZE = 128
API_BATCH_SIZE = 8 # max 500 symbols
# SIZE = 100

eng_train_dataset_filepath = f'{SOURCE_ROOT}/train.csv'
eng_val_dataset_filepath = f'{SOURCE_ROOT}/val.csv'
eng_test_dataset_filepath = f'{SOURCE_ROOT}/test.csv'
eng_answer_space_filepath = f'{SOURCE_ROOT}/answer_space.txt'

ukr_train_dataset_filepath = f'{OUTPUT_ROOT}/train.csv'
ukr_val_dataset_filepath = f'{OUTPUT_ROOT}/val.csv'
ukr_test_dataset_filepath = f'{OUTPUT_ROOT}/test.csv'
ukr_answer_space_filepath = f'{OUTPUT_ROOT}/answer_space.txt'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

model = MarianMTModel.from_pretrained(MODEL_NAME).to(device)
tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)


class CustomTranslator:
    def __init__(self, input_filepath: str, output_filepath: str, has_answers: bool, use_google_translator: bool) -> None:
        self.filepath = input_filepath
        self.output_filepath = output_filepath
        self.has_answers = has_answers
        self.use_google_translator = use_google_translator

        self.data = pd.read_csv(input_filepath)

        self.eng_questions = self.data['question'].copy().to_list()
        self.eng_answers = self.data['answer'].copy().to_list() if self.has_answers else []
        self.ukr_questions = []
        self.ukr_answers = [] if self.use_google_translator else self.eng_answers
        self.ukr_images = self.data['image']

        self.translate_eng_to_ukr_using_model()
        # self.translate_eng_to_ukr_using_translator()

        self.save_csv()


    def save_csv(self):
        print(f'{len(self.ukr_questions)=}, {len(self.ukr_answers)=}, {len(self.ukr_images)=}')

        if self.has_answers:
           ukr_dataframe = pd.DataFrame({
                'question': self.ukr_questions,
                'answer': self.ukr_answers,
                'image': self.ukr_images
            })
        else:
            ukr_dataframe = pd.DataFrame({
                'question': self.ukr_questions,
                'image': self.ukr_images
            })
        ukr_dataframe.to_csv(self.output_filepath, index=False)


    def translate_eng_to_ukr_using_model(self):
        # Translate the text in batches
        for i in tqdm(range(0, len(self.eng_questions), MODEL_BATCH_SIZE)):
            # Tokenize the batch of input text
            batch = self.eng_questions[i:i+MODEL_BATCH_SIZE]
            batch_tokens = tokenizer.batch_encode_plus(
                batch, 
                return_tensors='pt',
                padding='longest',
                truncation=True
            ).to(device)

            # Perform translation
            translated_tokens = model.generate(**batch_tokens)

            # Decode the translated tokens
            translated_batch = [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in translated_tokens]
            self.ukr_questions.extend(translated_batch)

    
    def translate_eng_to_ukr_using_translator(self):
        for i in tqdm(range(0, len(self.eng_answers), API_BATCH_SIZE)):
            eng_words_list = self.eng_answers[i:i+API_BATCH_SIZE].copy().tolist()
            ukr_words_list = eng_words_list.copy()

            eng_indexes = []
            eng_words_subset = []
            for j, word in enumerate(eng_words_list):
                if not isinstance(word, str):
                    ukr_words_list[j] = word
                elif word.lower() == 'yes':
                    ukr_words_list[j] = 'так'
                elif word.lower() == 'no':
                    ukr_words_list[j] = 'ні'
                else:
                    eng_indexes.append(j)
                    eng_words_subset.append(word.lower())

            print(f'{eng_words_subset=}')

            if eng_words_subset:
                eng_words = ', '.join(eng_words_subset)
                ukr_words = translator.translate(eng_words)
                ukr_words = ukr_words.split(', ')

                for j, word in enumerate(ukr_words):
                    ukr_words_list[eng_indexes[j]] = word

            if len(ukr_words_list) != len(eng_words_list):
                print(f'{len(ukr_words_list)=}, {len(eng_words_list)=}')
                print(f'{ukr_words_list=}')
                print(f'{eng_words_list=}')

            self.ukr_answers.extend(ukr_words_list)

# train
custom_translator = CustomTranslator(
    input_filepath=eng_train_dataset_filepath, 
    output_filepath=ukr_train_dataset_filepath,
    has_answers=True,
    use_google_translator=False
)

# val
custom_translator = CustomTranslator(
    input_filepath=eng_val_dataset_filepath, 
    output_filepath=ukr_val_dataset_filepath,
    has_answers=True,
    use_google_translator=False
)

# test
custom_translator = CustomTranslator(
    input_filepath=eng_test_dataset_filepath, 
    output_filepath=ukr_test_dataset_filepath,
    has_answers=False,
    use_google_translator=False
)
