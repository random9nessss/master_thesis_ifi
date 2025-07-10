import os
import time
import json

from apiclients.deepseek import DeepseekClient
from datagenerator.email_generator import EmailGenerator
from attributes.email_attribute_sampler import AttributeSampler
from quotegenerator.quote_generator import FreightQuoteEngine
from config.logger import CustomLogger

logger = CustomLogger(name="Attr. Prompting Deepseek")


def main():
    total_samples = int(input("Enter total number of samples to generate: "))
    max_batch_size = 50

    sampler = AttributeSampler(seed=None)
    distance_matrix_path = os.path.join(os.getcwd(), "datasets_processed", "distance_matrix.xlsx")
    fq_engine = FreightQuoteEngine(distance_matrix_path)
    email_gen = EmailGenerator(sampler, fq_engine)

    deepseek_client = DeepseekClient(email_gen)

    batch_number = 1

    # -------------------------------------------------------------------
    # Iterative Data Generation
    # -------------------------------------------------------------------
    while total_samples > 0:
        batch_size = min(total_samples, max_batch_size)
        logger.info(f"Submitting batch #{batch_number} with {batch_size} samples...")

        for i in range(batch_size):
            response = deepseek_client.get_chat_completion()
            response_clean = deepseek_client._clean_sample(response)
            deepseek_client.serialize_response(response_clean)
            time.sleep(1)

        # -------------------------------------------------------------------
        # Result Aggregation
        # -------------------------------------------------------------------
        deepseek_client.aggregate_results()

        total_samples -= batch_size
        batch_number += 1

    logger.ok("All batches submitted and completed.")


if __name__ == "__main__":
    main()
