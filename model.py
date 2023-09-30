import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import plotly.express as px
from huggingface_hub import hf_hub_download
from config import app_config

### Reproducibility
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def load_model():
    ### useful variable that tells us whether we should use the GPU
    use_cuda = torch.cuda.is_available()

    ### Our model is based on VGG19 with classifier head replaced with custom layers
    ### Load the pretrained VGG19 model
    model_transfer = models.vgg19(weights="DEFAULT")

    ### Instantiate a custom classifier for our 50 classes and use it with pretrained model
    clf = nn.Sequential(
        nn.Linear(in_features=25088, out_features=4096),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=4096, out_features=1024),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=1024, out_features=50),
        nn.LogSoftmax(dim=1),
    )
    model_transfer.classifier = clf
    if use_cuda:
        model_transfer = model_transfer.cuda()

    ### Download the saved fine-tuned model weights and load it
    model_weights = hf_hub_download(
        repo_id=app_config.hf_repo_id, filename=app_config.hf_weights_file
    )
    if not use_cuda:
        state_dict = torch.load(model_weights, map_location=torch.device("cpu"))
    else:
        state_dict = torch.load(model_weights)
    model_transfer.load_state_dict(state_dict)

    return model_transfer


def predict(img, k):
    model = app_config.model
    classes = app_config.classes
    mean = torch.tensor(app_config.mean)
    std = torch.tensor(app_config.std)

    # Prepare image for model prediction
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    img = transform(img).unsqueeze(dim=0)

    # Predict
    model.eval()
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        img = img.cuda()
    pred = model(img)  # LogSoftmax output
    pred_proba = torch.exp(pred)  # Prediction probabilities
    proba, labels = torch.topk(pred_proba, k)
    labels = labels.cpu().numpy().squeeze()
    landmarks = [classes[i] for i in labels]
    # Cleanup the labels
    landmarks = [" ".join(label.split("_")) for label in landmarks]
    proba = proba.squeeze().tolist()
    fig = px.bar(
        pd.DataFrame(data={"Landmarks": landmarks, "Probabilities": proba}),
        y="Probabilities",
        x="Landmarks",
    )
    return landmarks, proba, fig
