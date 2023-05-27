import ssl
import uvicorn
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware

from typing import Dict
import yaml
import constants
from utils import loadAnswerSpace, featurizeImage, tokenizeQuestion
from model import MultimodalVQAModel

import torch
import os
from PIL import Image

app = FastAPI(ssl_keyfile="./certificates/key.pem", ssl_certfile="./certificates/cert.pem")
app.config: dict = None

origins = [
    "https://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def startup() -> None:
    """
    Load config, model & answer space.
    
    Returns:
    None
    """

    # load config file
    with open(constants.CONFIG_FILEPATH, 'r') as file:
        app.config = yaml.load(file, yaml.FullLoader)

    # load the vocabulary of all answers
    app.answer_space = loadAnswerSpace(
        dataset_root_dir=app.config['dataset_root_dir'],
        dataset_dir=app.config['dataset_dir'],
        answer_space_filename=app.config['answer_space_filename']
    )

    # load model
    app.model = MultimodalVQAModel(
        pretrained_text_name=app.config['text_model'],
        pretrained_image_name=app.config['image_model'],
        num_labels=len(app.answer_space),
        intermediate_dim=app.config['intermediate_dim']
    )

    # load a ckeckpoint
    app.model.load_state_dict(torch.load(os.path.join(".", "checkpoint", app.config['model_folder'], app.config['checkpoint'], "pytorch_model.bin")))
    app.model.to(app.config['device'])

    # switch model to the evaluation mode
    app.model.eval()


@app.on_event('startup')
async def startup_event() -> None:
    """
    Registers a startup function.
    
    Returns:
    None
    """

    await startup()


@app.post("/predict")
async def predict(image: UploadFile = Form(...), question: str = Form(...)) -> Dict[str, str]:
    """
    Save the image in the /usr directory, make predictions on the given (image, question) pair.
    
    Parameters:
    image (UploadFile): image for the VQA
    question (question): question for the VQA

    Returns:
    Dict[str, str]: predicted answer on the VQA task
    """

    # open a given image
    uploaded_image = Image.open(image.file)

    # save the given image
    image_path = f'./usr/{image.filename}'
    uploaded_image.save(image_path)

    # tokenize the question
    question = question.lower().replace("?", "").strip()
    tokenized_question = tokenizeQuestion(app.config['text_model'], question, app.config['device'])

    # featureize the image
    featurized_img = featurizeImage(app.config['image_model'], image_path, app.config['device'])

    # move features to the device
    input_ids = tokenized_question["input_ids"].to(app.config['device'])
    token_type_ids = tokenized_question["token_type_ids"].to(app.config['device'])
    attention_mask = tokenized_question["attention_mask"].to(app.config['device'])
    pixel_values = featurized_img["pixel_values"].to(app.config['device'])

    # obtain the prediction from the model
    output = app.model(input_ids, pixel_values, attention_mask, token_type_ids)

    # obtain the answer from the answer space
    preds = output["logits"].argmax(axis=-1).cpu().numpy()
    answer = app.answer_space[preds[0]]

    return {
        "prediction": answer
    }


# start the application
if __name__ == "__main__":
    uvicorn.run(
        "app:app", 
        host="127.0.0.1", 
        port=8000, 
        ssl_keyfile="./certificates/key.pem", 
        ssl_certfile="./certificates/cert.pem", 
        reload=True
    )
