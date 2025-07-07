import requests

from openai import OpenAI
from utils import get_kaggle, search_web
from utils.embeddings import chunk_and_retrieve
from configs import AVAILABLE_LLMs
from validators import url


def retrieve_models(user_requirements):
    """Retrieve up-to-date and state-of-the-art knowledge from the websearch and relevant hubs for modeling and optimization."""

    # Return detailed sources and implementation for subsequent code generation
    models = []

    for model in user_requirements["model"]:
        # find match with reducing priority
        model["modality"] = []
        for dataset in user_requirements["dataset"]:
            model["modality"] += dataset["modality"]
        model["modality"] = set(model["modality"])

        found_model = retrieve_huggingface(**model)
        # found_model = "exist in source" if found_model else found_model
        if found_model:
            models.append(
                {
                    "name": model["name"],
                    "model": found_model,
                    "source": "huggingface-hub",
                }
            )
            continue

        found_model = retrieve_kaggle(**model)
        # formerly, tensorflow hub
        if found_model:
            models.append(
                {"name": model["name"], "model": found_model, "source": "kaggle-hub"}
            )
            continue

        found_model, hub_name = retrieve_pytorch(**model)
        if found_model:
            models.append(
                {"name": model["name"], "model": found_model, "source": hub_name}
            )
            continue

    return models


def retrieve_huggingface(**kwargs):
    from huggingface_hub import HfApi

    hf_api = HfApi()
    if kwargs["name"] and kwargs["name"] != "":
        model = list(
            hf_api.list_models(
                search=kwargs["name"], sort="downloads", full=True, cardData=True
            )
        )
        if len(model) > 0:
            return model[0]
        else:
            return False        
    elif kwargs["family"] and kwargs["family"] != "":
        model = list(
            hf_api.list_models(
                search=kwargs["family"], sort="downloads", full=True, cardData=True
            )
        )
        if len(model) > 0:
            return model[0]
        else:
            return False


def retrieve_kaggle(**kwargs):
    kaggle_api = get_kaggle()
    if kwargs["name"] and kwargs["name"] != "":
        model = kaggle_api.models_list(search=kwargs["name"], sort_by="voteCount")
        if len(model['models']) > 0:
            return model["models"][0]
        else:
            return False        
    elif kwargs["family"] and kwargs["family"] != "":
        model = kaggle_api.models_list(search=kwargs["family"], sort_by="voteCount")
        if len(model['models']) > 0:
            return model["models"][0]
        else:
            return False


def retrieve_pytorch(**kwargs):
    if kwargs["name"] and kwargs["name"] != "":
        query = kwargs["name"]
    elif kwargs["family"] and kwargs["family"] != "":
        query = kwargs["family"]
    else:
        return False, False

    utility_classes = ["to_captum_model", "to_captum_input", "captum_output_to_dicts"]
    if "image" in kwargs["modality"] or "video" in kwargs["modality"]:
        from torchvision import models

        avail_models = [
            fn.lower().replace("-", " ").replace("_", " ")
            for fn in models.list_models()
            if fn not in utility_classes
        ]
        hub_name = "torchvision"
    elif "text" in kwargs["modality"]:
        from torchtext import models

        model_names = [
            model for model in list(models.__dict__.keys()) if "__" not in model
        ]
        avail_models = [
            fn.lower().replace("-", " ").replace("_", " ")
            for fn in model_names
            if fn not in utility_classes
        ]

        hub_name = "torchtext"
    elif "audio" in kwargs["modality"]:
        from torchaudio import models

        avail_models = [
            fn.lower().replace("-", " ").replace("_", " ")
            for fn in models.__all__
            if fn not in utility_classes
        ]

        hub_name = "torchaudio"
    elif "graph" in kwargs["modality"]:
        from torch_geometric.nn import models

        avail_models = [
            fn.lower().replace("-", " ").replace("_", " ")
            for fn in models.__all__
            if fn not in utility_classes
        ]

        hub_name = "torch_geometric"
    else:
        # not support multimodal, time series, and tabular modalities
        return False, False

    query = query.lower().replace("-", " ").replace("_", " ")
    if query in avail_models:
        return avail_models[avail_models.index(query)], hub_name
    else:
        return False, False
