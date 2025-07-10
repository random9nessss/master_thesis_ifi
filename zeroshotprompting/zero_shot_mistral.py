import os
import time
import json
from datetime import datetime
from zeroshotprompting.zeroshot_mistral import MistralZeroShotClient
from config.logger import CustomLogger

logger = CustomLogger(name="Zero Shot Prompting Mistral")


def wait_for_all_jobs(client):
    """
    Wait for all active batch jobs to complete using an exponential backoff strategy.

    This function continuously monitors the status of active batch jobs by invoking
    the client's `monitor_batch_jobs()` method and checking the contents of the
    'active_batch_jobs.json' file located in the client's directory. It employs an
    exponential backoff approach—starting with a wait time of 200 seconds and doubling
    the wait time (up to a maximum of 2000 seconds) after each check—until no active jobs
    remain.

    Args:
        client: An instance of the batch job client (MistralClient) which must have:
            - _directory (str): The directory where job files are stored.
            - monitor_batch_jobs() (callable): A method that checks and updates job statuses.
            - logger: A logger instance for logging messages.

    Returns:
        None
    """
    wait_time = 200
    while True:
        client.monitor_batch_jobs()
        active_jobs_file = os.path.join(client._directory, "active_batch_jobs.json")
        active_jobs = []
        if os.path.exists(active_jobs_file):
            try:
                with open(active_jobs_file, "r", encoding="utf-8") as f:
                    active_jobs = json.load(f)
            except json.JSONDecodeError:
                active_jobs = []
        if not active_jobs:
            break

        logger.info(f"Active jobs still running: {active_jobs}. Waiting for {wait_time} seconds...")
        time.sleep(wait_time)
        wait_time = min(wait_time * 2, 2000)


def main():
    total_samples = int(input("Enter total number of samples to generate: "))
    max_batch_size = 500

    mistral_client = MistralZeroShotClient(mode="fewshot")
    batch_number = 1

    # -------------------------------------------------------------------
    # Iterative Data Generation
    # -------------------------------------------------------------------
    while total_samples > 0:
        batch_size = min(total_samples, max_batch_size)
        logger.info(f"Submitting batch #{batch_number} with {batch_size} samples...")
        mistral_client.run_batch_inference(number_samples=batch_size)
        wait_for_all_jobs(mistral_client)
        total_samples -= batch_size
        batch_number += 1

    logger.ok("All batches submitted and completed.")

if __name__ == "__main__":
    main()