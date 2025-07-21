"""
curl -v -X POST http://127.0.0.1:5000/compute_embedding -H "Content-Type: application/json" -d '{"sentences": ["Hello world!"], "model_name": "gte-large-en-v1.5"}'

"""

import os
import sys
import time
import numpy as np
import json

# import threading
import logging
import torch
import onnxruntime as ort

# import intel_extension_for_pytorch as ipex
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

import subprocess
import requests


MODEL_NAMES = {
    "TEI": [
        "multilingual-e5-large-instruct",
        # "UAE-Large-V1",
    ],
    "sentence_transformer": [
        # "SBERT-bert-base-spanish-wwm-uncased",
        # "LaBSE",
        # "sentence-camembert-large",  # fr
    ],
    "huggingface": [],
    "pytorch": [],
}

# Configure root logger to handle INFO messages
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Suppress specific logging warnings
logging.getLogger("werkzeug").setLevel(logging.ERROR)
logging.getLogger("IPEX").setLevel(logging.ERROR)

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

MODEL_ZOO_DIR = "model_zoo"  # Go up one directory from embed/ to access model_zoo/
# To serve a model, create an empty file with the model name in this directory
MODEL_NAME_DIR = "loaded_model_names"
os.makedirs(MODEL_NAME_DIR, exist_ok=True)
CHECK_INTERVAL = 2  # Check every N seconds for the list of model names

BATCH_SIZE = 64

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# We may need multiple models at the same time, so we use a dictionary to store them
# {model_name: model_instance}
model_dict = {}


def model_type(model_name: str) -> str:
    """
    Determine the type of model based on its name.
    """
    if model_name in MODEL_NAMES["TEI"]:
        return "TEI"
    elif model_name in MODEL_NAMES["sentence_transformer"]:
        return "sentence_transformer"
    elif model_name in MODEL_NAMES["huggingface"]:
        return "huggingface"
    elif model_name in MODEL_NAMES["pytorch"]:
        return "pytorch"
    else:
        return "other"


def _load_model_sentence_transformer(
    model_name: str, warmup: bool = True
) -> SentenceTransformer:
    start = time.time()
    os.makedirs(MODEL_ZOO_DIR, exist_ok=True)
    print(f"Instantiating model: {model_name}")

    model = SentenceTransformer(
        os.path.join(MODEL_ZOO_DIR, model_name), trust_remote_code=True
    )

    model = model.to(DEVICE)
    # model = torch.compile(model, backend="torch_tensorrt", dynamic=False,
    #                         options={
    #                             "truncate_long_and_double": True,
    #                             "precision": torch.half
    #                             }
    #                         )
    model.eval()

    # if not full_precision:
    #     print(f'Optimizing')
    #     model = ipex.optimize(model, dtype=torch.bfloat16)

    print(f"Done loading model: {model_name}. Took {time.time() - start:.2f}s")
    if warmup:
        start = time.time()
        with torch.inference_mode():
            model.encode(
                [
                    "Artificial intelligence (AI), in its broadest sense, is intelligence exhibited by machines, particularly computer systems. It is a field of research in computer science that develops and studies methods and software that enable machines to perceive their environment and use learning and intelligence to take actions that maximize their chances of achieving defined goals. Such machines may be called AIs."
                ]
            )
        # with torch.inference_mode(), torch.cpu.amp.autocast():
        #     model.encode(['Hello'])
        print(f"Warmup took an extra {time.time() - start:.2f}s")
    return model


class HFONNXModel:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def encode(self, sentences, batch_size=64):
        if isinstance(sentences, str):
            sentences = [sentences]
        all_embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i : i + batch_size]
            inputs = self.tokenizer(
                batch_sentences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = self._mean_pooling(
                    outputs.last_hidden_state, inputs["attention_mask"]
                )
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings)
        if all_embeddings:
            return torch.cat(all_embeddings, dim=0).cpu().numpy()
        else:
            return np.array([])

    def _mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )


def _load_model_hfonnx(
    model_name: str, warmup: bool = True, onnx_filename: str = "model.onnx"
) -> HFONNXModel:
    """
    Load a Hugging Face Transformer model with ONNX weights for optimized inference.
    Depending on the model, multiple ONNX models may be available:
        - model.onnx (standard)
        - model_quantized.onnx (quantized for smaller size/faster inference)
        - model_fp16.onnx (half-precision for reduced memory usage)
    """
    start = time.time()
    os.makedirs(MODEL_ZOO_DIR, exist_ok=True)
    print(f"Instantiating model: {model_name}")

    model_path = os.path.join(MODEL_ZOO_DIR, model_name)
    onnx_path = os.path.join(model_path, "onnx")

    # Try loading ONNX model using optimum.onnxruntime
    try:
        # Determine available providers
        available_providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in available_providers and DEVICE == "cuda":
            provider = "CUDAExecutionProvider"
        else:
            provider = "CPUExecutionProvider"

        model = ORTModelForFeatureExtraction.from_pretrained(
            onnx_path,
            file_name=onnx_filename,
            provider=provider,
            library="transformers",
        )
        print("Successfully loaded ONNX model using optimum.onnxruntime")
    except Exception as e:
        print(f"Optimum ONNX load failed: {e}")
        # Fallback to direct onnxruntime
        try:
            onnx_model_path = os.path.join(onnx_path, "model.onnx")
            available_providers = ort.get_available_providers()
            if "CUDAExecutionProvider" in available_providers and DEVICE == "cuda":
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
            ort_session = ort.InferenceSession(onnx_model_path, providers=providers)

            class ONNXRuntimeWrapper:
                def __init__(self, session):
                    self.session = session
                    self.input_names = [inp.name for inp in session.get_inputs()]
                    self.output_names = [out.name for out in session.get_outputs()]

                def __call__(self, **inputs):
                    onnx_inputs = {
                        name: (value.cpu().numpy() if hasattr(value, "cpu") else value)
                        for name, value in inputs.items()
                    }
                    outputs = self.session.run(self.output_names, onnx_inputs)
                    result = type("obj", (object,), {})()
                    result.last_hidden_state = torch.from_numpy(outputs[0])
                    return result

            model = ONNXRuntimeWrapper(ort_session)
            print("Successfully loaded ONNX model using direct onnxruntime")
        except Exception as e2:
            print(f"Direct ONNX loading failed: {e2}")
            # Fallback to regular transformers
            from transformers import AutoModel

            model = AutoModel.from_pretrained(model_path)
            model = model.to(DEVICE)
            model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print(
        f"Done loading HuggingFace ONNX model: {model_name}. Took {time.time() - start:.2f}s"
    )

    if warmup:
        start = time.time()
        inputs = tokenizer(
            "Artificial intelligence (AI), in its broadest sense, is intelligence exhibited by machines.",
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        with torch.no_grad():
            _ = model(**inputs)
        print(f"Warmup took an extra {time.time() - start:.2f}s")

    return HFONNXModel(model, tokenizer)


def _load_model_torch(model_name: str, warmup: bool = True) -> None:
    return None


def load_model(model_name: str, warmup: bool = True):
    """
    Load a model from the model zoo directory.
    """
    if model_type(model_name) == "TEI":
        model = None
    elif model_type(model_name) == "sentence_transformer":
        model = _load_model_sentence_transformer(model_name, warmup=warmup)
    elif model_type(model_name) == "huggingface":
        model = _load_model_hfonnx(model_name, warmup=warmup)
    elif model_type(model_name) == "pytorch":
        model = _load_model_torch(model_name, warmup=warmup)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


def check_for_model_update():
    global model_dict
    while True:
        # Check for the updated model names
        # The folder contains files named {call_sid}___{model_name}
        new_model_names = os.listdir(MODEL_NAME_DIR)
        new_model_names = set([x.split("___")[-1] for x in new_model_names])
        # Remove old models that are no longer in the directory
        for model_name in list(model_dict.keys()):
            if model_name not in new_model_names:
                print(f"Removing model: {model_name}")
                del model_dict[model_name]
        # Load new models
        for model_name in new_model_names:
            if model_name not in model_dict:
                print(f"Loading model: {model_name}")
                model_dict[model_name] = load_model(model_name)
        time.sleep(CHECK_INTERVAL)


def _compute_embeddings_tei(sentences: list[str], model_name: str, batch_size=32) -> list:
    # TEI service URL (port 8080, not 5000 like your Flask server)
    url = f"http://127.0.0.1:8080/{model_name}"
    headers = {"Content-Type": "application/json"}
    
    # TEI expects the data in this format: {"inputs": ["sentence1", "sentence2", ...]}
    data = json.dumps({"inputs": sentences})
    
    # Make the POST request to TEI service
    response = requests.post(url, headers=headers, data=data)
    
    # Parse the response
    if response.status_code == 200:
        embeddings = response.json()
        return embeddings
    else:
        raise Exception(f"TEI request failed with status {response.status_code}: {response.text}")
        


def compute_embeddings(sentences: list[str], model_dict: dict, model_name: str) -> list:
    """
    Compute embeddings for a list of sentences using the specified model.
    """
    embeddings = None
    model = model_dict[model_name]

    if model_type(model_name) == "TEI":
        embeddings = _compute_embeddings_tei(sentences, model_name) # shape (1, 1024) 
    elif model_type(model_name) == "sentence_transformer":
        with torch.inference_mode():
            embeddings = model.encode(sentences).tolist() # shape (1,768)
    elif model_type(model_name) == "huggingface":
        pass
    elif model_type(model_name) == "pytorch":
        pass
    else:
        pass

    return embeddings


@app.route("/compute_embedding", methods=["POST"])
def predict():
    if not model_dict:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json()
    # app.logger.info(f"Received data: {data}")
    sentences = data["sentences"]
    model_name = data["model_name"]
    if not sentences:
        app.logger.warning("Received no sentences.")
        return jsonify({"error": "No sentences provided"}), 400

    app.logger.info(
        f'Received request: model_name: {model_name}, {len(sentences)} sentence(s): "{sentences[0]}",...'
    )

    if model_name not in model_dict:
        print(f"*** WARNING: Model {model_name} not already loaded. Loading it now...")
        model_dict[model_name] = load_model(model_name)

    # with torch.inference_mode(), torch.autocast(device_type=DEVICE, dtype=torch.float16):
    #     embeddings = model_dict[model_name].encode(sentences, batch_size=BATCH_SIZE)

    # with torch.inference_mode():
    #     embeddings = model_dict[model_name].encode(sentences, batch_size=BATCH_SIZE)
    

    # embeddings = embeddings.tolist()

    embeddings = compute_embeddings(sentences, model_dict, model_name)

    return jsonify(embeddings), 200


def run_app():
    # Start the model update checker in a separate thread
    # threading.Thread(target=check_for_model_update, daemon=True).start()
    print(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', None)}")

    # Start the embedding network for TEI models in the background
    script_dir = os.path.dirname(os.path.abspath(__file__))
    embed_start_path = os.path.join(script_dir, "embed_start.sh")
    subprocess.Popen(["bash", embed_start_path])

    # Load all non TEI models
    # model_names = ["gte-large-en-v1.5", "sentence-camembert-large", "all-MiniLM-L6-v2"]
    model_names = (
        MODEL_NAMES["TEI"]
        + MODEL_NAMES["sentence_transformer"]
        + MODEL_NAMES["huggingface"]
    )
    # Load all models to the global model_dict
    for model_name in model_names:
        model_dict[model_name] = load_model(model_name, warmup=False)
    app.run(host="0.0.0.0", port=5000)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        assert sys.argv[1].lower() in ["daemon_true", "daemon_false"]
        if sys.argv[1].lower() == "daemon_true":
            daemon_exec = "yes"
        if sys.argv[1].lower() == "daemon_false":
            daemon_exec = "no"
    else:
        daemon_exec = "no"

    os.environ["daemon_exec"] = daemon_exec
    run_app()


