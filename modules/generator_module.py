from . import *
from .modules_logger import *

class GeneratorModule:
    def __init__(self, config):
        self._generator = get_generator(config)
        logger.debug("Generator model loaded")

    def __call__(self, inputs):
        """
        Generate answers for the given inputs using the generator instance.
        Replaces empty answers with a default "No answer found" message.

        :param inputs: list of input queries or prompts
        :return: tuple of (list of answers, elapsed_time_in_seconds)
        """
        logger.debug("Generation phase started")
        start = time.time()

        pred_answer_list = self._generator.generate(inputs)

        end = time.time()
        logger.debug("Generation phase ended")

        return pred_answer_list, end - start