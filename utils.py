import os
import transformers
from typing import List, Dict, Union
from pathlib import Path
from PIL import Image


def loadAnswerSpace(dataset_root_dir: str, dataset_dir: str, answer_space_filename: str) -> List[str]:
    """
    Loads the answer space from a saved dictionary.
    
    Parameters:
    dataset_root_dir (str): root dataset directory
    dataset_dir (str): certain dataset directory
    answer_space_filename (str): answer space filename

    Returns:
    answer_space List[str]: list of all possible answers
    """

    with open(os.path.join(dataset_root_dir, dataset_dir, answer_space_filename)) as f:
        answer_space = f.read().splitlines()
    return answer_space


def tokenizeQuestion(text_encoder: str, question: str, device: str) -> Dict:
    """
    Tokenize a question using the given model.

    Parameters:
    text_encoder (str): model name
    question (str): asked question
    device (str): device for the question features

    Returns:
    Dict: tokenized question
    """

    tokenizer = transformers.AutoTokenizer.from_pretrained(text_encoder)
    encoded_text = tokenizer(
        text=[question],
        padding='longest',
        max_length=24,
        truncation=True,
        return_tensors='pt',
        return_token_type_ids=True,
        return_attention_mask=True,
    )
    return {
        "input_ids": encoded_text['input_ids'].to(device),
        "token_type_ids": encoded_text['token_type_ids'].to(device),
        "attention_mask": encoded_text['attention_mask'].to(device),
    }


def featurizeImage(image_encoder: str, img_path: Union[Path, str], device: str) -> Dict:
    """
    Get features for a given image.

    Parameters:
    image_encoder (str): model name
    img_path Union[Path, str]: given image filepath
    device (str): device for the processed image

    Returns:
    Dict: processed image pixels
    """

    featurizer = transformers.AutoFeatureExtractor.from_pretrained(image_encoder)
    processed_images = featurizer(
            images=[Image.open(img_path).convert('RGB')],
            return_tensors="pt",
        )
    return {
        "pixel_values": processed_images['pixel_values'].to(device),
    }
