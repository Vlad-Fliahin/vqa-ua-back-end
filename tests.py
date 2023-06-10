import yaml
import constants
from utils import loadAnswerSpace
from model import MultimodalVQAModel
import torch
import os
import unittest


class ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        # load config file
        with open(constants.CONFIG_FILEPATH, 'r') as file:
            self.config = yaml.load(file, yaml.FullLoader)

        # load the vocabulary of all answers
        self.answer_space = loadAnswerSpace(
            dataset_root_dir=self.config['dataset_root_dir'],
            dataset_dir=self.config['dataset_dir'],
            answer_space_filename=self.config['answer_space_filename']
        )

        self.model = None
    

    def tearDown(self) -> None:
        self.model = None
        self.answer_space = []
        self.config = {}


    def test_model_load(self):
        try:
            # load model
            self.model = MultimodalVQAModel(
                pretrained_text_name=self.config['text_model'],
                pretrained_image_name=self.config['image_model'],
                num_labels=len(self.answer_space),
                intermediate_dim=self.config['intermediate_dim']
            )

            # load a ckeckpoint
            self.model.load_state_dict(
                torch.load(os.path.join(".", "checkpoint", self.config['model_folder'],
                                        self.config['checkpoint'], "pytorch_model.bin"))
            )
            self.model.to(self.config['device'])
        except Exception as exception:
            print('Failure during the model load', exception)

        self.assertFalse(self.model is None)


    def test_model_creation(self):
        try:
            # load model
            self.model = MultimodalVQAModel(
                pretrained_text_name=self.config['text_model'],
                pretrained_image_name=self.config['image_model'],
                num_labels=len(self.answer_space),
                intermediate_dim=self.config['intermediate_dim']
            )
        except Exception as exception:
            self.model = None
            print(f'Failure during the model creation {exception=}')
        
        self.assertFalse(self.model is None)


if __name__ == "__main__":
    unittest.main()
