from . import *
from .modules_logger import *

class RefinerModule:
    def __init__(self, config):
        self._refiner = get_refiner(config)

    def __call__(self, dataset, template, prompts):
        """
        Runs the document refiner phase on the dataset.

        If the refiner's name includes 'llmlingua' and the input_prompt_flag is set,
        it uses the input prompts directly for refinement.

        Otherwise, it refines retrieval documents and formats prompts using the provided template.

        :param dataset: dataset object containing questions and data
        :param template: a prompt template with get_string() method
        :param prompts: list of prompt strings (used if input_prompt_flag is True)
        :return: tuple of (list of input prompts, elapsed time in seconds)
        """
        logger.debug("Refiner phase started")
        start = time.time()

        input_prompt_flag = self._refiner.input_prompt_flag

        if "llmlingua" in self._refiner.name and input_prompt_flag:
            # Use input prompts directly for refinement
            dataset["prompt"] = prompts
            input_prompts = self._refiner.batch_run(dataset)
        else:
            # Refine retrieval documents and generate formatted prompts
            refine_results = self._refiner.batch_run(dataset)
            input_prompts = [
                template.get_string(question=q, formatted_reference=r)
                for q, r in zip(dataset.question, refine_results)
            ]

        end = time.time()
        logger.debug("Refiner phase ended")

        return input_prompts, end - start