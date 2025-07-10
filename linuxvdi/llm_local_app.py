import requests
from huggingface_hub import configure_http_backend
from vllm import LLM, SamplingParams
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv

import argparse
import uvicorn

import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger().setLevel(logging.WARNING)
logging.getLogger("vllm").setLevel(logging.ERROR)
logging.getLogger("llm_engine").setLevel(logging.ERROR)
logging.getLogger("cuda").setLevel(logging.ERROR)
logging.getLogger("multiproc_worker_utils").setLevel(logging.ERROR)
logging.getLogger("custom_all_reduce").setLevel(logging.ERROR)
logging.getLogger("custom_cache_manager").setLevel(logging.ERROR)

def backend_factory() -> requests.Session:
    """
     Create and configure a Requests session that disables SSL verification.

     Returns:
         requests.Session: A configured Requests session.
     """
    session = requests.Session()
    session.verify = False
    return session

configure_http_backend(backend_factory=backend_factory)

class LocalClient:
    """
    A local client for hosting a vLLM model and interacting with it.

    This class loads the specified Hugging Face model with vLLM and exposes a method to
    generate text based on an input prompt.

    Attributes:
        model (str): The Hugging Face model identifier.
        tensor_parallel_size (int): Number of GPUs for tensor parallelism.
        max_tokens (int): Maximum tokens to generate.
        temperature (float): Sampling temperature.
        top_p (float): Top-p (nucleus) sampling parameter.
        max_seq_length (int): Maximum sequence length for model context.
        gpu_memory_utilization (float): Fraction of GPU memory to use.
    """
    def __init__(self,
                 model: str,
                 tensor_parallel_size: int = 2,
                 max_tokens: int = 250,
                 temperature: float = 1.0,
                 top_p: float = 0.9,
                 max_seq_length: int = 2048,
                 gpu_memory_utilization: float = 0.8):
        """
        Initialize the LocalClient with the specified model and sampling parameters.

        Args:
            model (str): The Hugging Face model identifier.
            tensor_parallel_size (int, optional): GPUs for tensor parallelism. Defaults to 2.
            max_tokens (int, optional): Maximum tokens to generate. Defaults to 5000.
            temperature (float, optional): Sampling temperature. Defaults to 0.9.
            top_p (float, optional): Top-p sampling parameter. Defaults to 0.95.
            max_seq_length (int, optional): Maximum context length. Defaults to 2048.
            gpu_memory_utilization (float, optional): Fraction of GPU memory to utilize. Defaults to 0.8.

        Raises:
            ValueError: If the Hugging Face token (HF_TOKEN) is not found in ENV.txt.
        """
        self.model = model
        self.tensor_parallel_size = tensor_parallel_size
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.max_seq_length = max_seq_length
        self.gpu_memory_utilization = gpu_memory_utilization

        # -------------------------------------------------------------------
        # Loading Huggingface Token
        # -------------------------------------------------------------------
        load_dotenv("ENV.txt")
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN not found in ENV.txt")

        # -------------------------------------------------------------------
        # Loading Model
        # -------------------------------------------------------------------
        self._load_llm()

    def _get_sampling_attributes(self) -> SamplingParams:
        """
        Construct and return the sampling parameters for text generation.

        Returns:
            SamplingParams: Parameters including temperature, top_p, and max_tokens.
        """
        return SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )

    def _load_llm(self) -> None:
        """
         Load the vLLM model with the specified parameters.

         This method initializes the model along with its sampling attributes.
         """
        self._sampling_attributes = self._get_sampling_attributes()
        self._llm = LLM(
            self.model,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_seq_length,
            dtype="float16"
        )

    def interact(self, prompt: str) -> str:
        """
        Generate a response for the provided prompt using the pre-loaded vLLM model.

        Args:
            prompt (str): The input text prompt.

        Returns:
            str: The concatenated prompt and the model-generated output.
        """
        output = self._llm.generate(prompt, sampling_params=self._sampling_attributes)
        return " ".join([output[0].prompt, output[0].outputs[0].text])

# -------------------------------------------------------------------
# Mapping Model Selection
# -------------------------------------------------------------------
MODEL_MAPPING = {
    "llama8b": "meta-llama/Llama-3.1-8B",
    "llama3b": "meta-llama/Llama-3.2-3B",
    "deepseek": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
}


class GenerateRequest(BaseModel):
    """
    Request body model for text generation.

    Attributes:
        prompt (str): The input prompt for the model.
    """
    prompt: str

# -------------------------------------------------------------------
# Fast API Application
# -------------------------------------------------------------------
app = FastAPI()

client = None

@app.on_event("startup")
async def startup_event():
    """
    Initialize the vLLM model when the API starts.

    This function reads the DEFAULT_MODEL from ENV.txt (or environment) and initializes
    the corresponding LocalClient instance. Defaults to 'llama8b' if not set.

    Raises:
        ValueError: If an invalid DEFAULT_MODEL is specified.
    """
    global client
    load_dotenv("ENV.txt")
    default_model_key = os.getenv("DEFAULT_MODEL", "llama8b").lower()

    if default_model_key not in MODEL_MAPPING:
        raise ValueError("Invalid DEFAULT_MODEL specified. Choose from 'llama8b', 'llama3b', or 'deepseek'.")

    selected_model = MODEL_MAPPING[default_model_key]
    client = LocalClient(model=selected_model)
    print(f"Model {selected_model} initialized.")

@app.post("/generate")
async def generate_text(request: GenerateRequest):
    """
    API endpoint to generate text using the pre-loaded model.

    Args:
        request (GenerateRequest): The request body containing the prompt.

    Returns:
        dict: A dictionary with the generated response text.

    Raises:
        HTTPException: If the model is not initialized or an error occurs during generation.
    """
    if client is None:
        raise HTTPException(status_code=500, detail="Model not initialized.")
    try:
        response_text = client.interact(request.prompt)
        return {"response": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FastAPI with a specified model")
    parser.add_argument("--model", type=str, choices=["llama8b", "llama3b", "deepseek"],
                        default="llama8b", help="Select the model to load")
    args = parser.parse_args()

    os.environ["DEFAULT_MODEL"] = args.model

    uvicorn.run(app, host="0.0.0.0", port=8000)