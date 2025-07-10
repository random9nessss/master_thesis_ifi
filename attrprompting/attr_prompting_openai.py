import os
import time
import json
from datetime import datetime

from apiclients.openai import OpenAIClient
from datagenerator.email_generator import EmailGenerator
from attributes.email_attribute_sampler import AttributeSampler
from quotegenerator.quote_generator import FreightQuoteEngine

from config.logger import CustomLogger

logger = CustomLogger(name="Attr. Prompting OpenAI")

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
        client: An object representing the client interfacing with the batch job API.
            It must have the following attributes:
                - _directory (str): The directory where batch job files (e.g.,
                  'active_batch_jobs.json') are stored.
                - monitor_batch_jobs() (callable): A method that updates the status of batch jobs.
                - logger: A logger instance with an `info` method for logging messages.

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
    max_batch_size = 50 # Batch API only permits queuing 90k input tokens

    sampler = AttributeSampler(seed=None)
    distance_matrix_path = os.path.join(os.getcwd(), "datasets_processed", "distance_matrix.xlsx")
    fq_engine = FreightQuoteEngine(distance_matrix_path)
    email_gen = EmailGenerator(sampler, fq_engine)
    openai_client = OpenAIClient(email_gen)

    batch_number = 1

    # -------------------------------------------------------------------
    # Iterative Data Generation
    # -------------------------------------------------------------------
    while total_samples > 0:
        batch_size = min(total_samples, max_batch_size)
        logger.info(f"Submitting batch #{batch_number} with {batch_size} samples...")
        batch_filename = f"batch_{datetime.now().strftime('%Y%m%d%H%M%S')}_{batch_number}.jsonl"
        openai_client.run_batch_inference(batch_filename, number_samples=batch_size)
        wait_for_all_jobs(openai_client)
        total_samples -= batch_size
        batch_number += 1

    logger.ok("All batches submitted and completed.")

if __name__ == "__main__":
    main()
