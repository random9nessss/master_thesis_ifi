import os
import time
import json

from apiclients.google import GoogleClient
from datagenerator.email_generator import EmailGenerator
from attributes.email_attribute_sampler import AttributeSampler
from quotegenerator.quote_generator import FreightQuoteEngine

from config.logger import CustomLogger

logger = CustomLogger(name="Attr. Prompting Google")

def main():
    total_samples = int(input("Enter total number of samples to generate: "))
    max_batch_size = 50

    sampler = AttributeSampler(seed=None)
    distance_matrix_path = os.path.join(os.getcwd(), "datasets_processed", "distance_matrix.xlsx")
    fq_engine = FreightQuoteEngine(distance_matrix_path)
    email_gen = EmailGenerator(sampler, fq_engine)
    google_client = GoogleClient(email_gen)

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
