import os
import time
import json

from zeroshotprompting.zeroshot_deepseek import DeepseekZeroShotClient
from config.logger import CustomLogger
logger = CustomLogger(name="Zero Shot Prompting Google")


def main():
    total_samples = int(input("Enter total number of samples to generate: "))
    deepseek_client = DeepseekZeroShotClient(mode="fewshot")
    deepseek_client.run_inference(number_samples=total_samples)
    logger.ok("All batches submitted and completed.")

if __name__ == "__main__":
    main()