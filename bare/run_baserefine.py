from bare.base import Base
from bare.refine import Refine

if __name__ == "__main__":

    # ------------------------------------------------------------------
    # Base
    # ------------------------------------------------------------------
    base = Base()
    base_jsonl_path = base.generate_email_chains_to_file(num_chains=50, file_name="base_chains.jsonl")

    # ------------------------------------------------------------------
    # Refine
    # ------------------------------------------------------------------
    refiner = Refine()
    refiner.create_refinement_tasks_from_jsonl(input_jsonl_file=base_jsonl_path, model="gpt-4-turbo", temperature=1.0)
    refiner.create_and_submit_batch(output_batch_filename="refinement_batch.jsonl")