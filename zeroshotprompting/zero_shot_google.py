import os
import time
import json

from zeroshotprompting.zeroshot_google import GoogleZeroShotClient
from config.logger import CustomLogger

logger = CustomLogger(name="Zero Shot Prompting Google")

def main():
    total_samples = int(input("Enter total number of samples to generate: "))
    max_batch_size = 50

    google_client = GoogleZeroShotClient(mode="fewshot")

    batch_number = 1

    # -------------------------------------------------------------------
    # Iterative Data Generation
    # -------------------------------------------------------------------
    while total_samples > 0:
        batch_size = min(total_samples, max_batch_size)
        logger.info(f"Submitting batch #{batch_number} with {batch_size} samples...")
        google_client.run_inference(number_samples=batch_size)
        total_samples -= batch_size
        batch_number += 1

    logger.ok("All batches submitted and completed.")

if __name__ == "__main__":
    main()