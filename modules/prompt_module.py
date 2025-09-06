from . import *
from .modules_logger import *

class PromptModule:
    def __init__(self, config, system_prompt=None, user_prompt=None):
        """
        Initializes the PromptModule with two types of prompts:
        - custom_prompt: uses a prompt that includes document references (retrieval results)
        - default_prompt: uses a standard prompt based only on internal knowledge

        :param config: configuration object used for PromptTemplate
        :param system_prompt: optional custom system prompt
        :param user_prompt: optional custom user prompt
        """
        if system_prompt:
            logger.debug(f"Using custom system prompt: {system_prompt}")

        system_prompt = system_prompt or (
            "Answer the question based on the given document. Only give me the answer and do not output any other words. "
            "The following are given documents:{reference}"
        )

        if user_prompt:
            logger.debug(f"Using custom user prompt: {user_prompt}")
        user_prompt = user_prompt or "Question: {question}\nAnswer:"

        self._custom_prompt = PromptTemplate(
            config,
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )

        self.default_prompt = PromptTemplate(
            config=config,
            system_prompt="Answer the question based on your own knowledge. Only give me the answer and do not output any other words.",
            user_prompt="Question: {question}\nAnswer:",
        )

    def generator_only_prompt(self, queries):
        """
        Generate a list of prompts based only on internal knowledge (default_prompt) for each query.

        :param queries: list of question strings
        :return: list of prompt strings
        """
        logger.debug(f"Using default prompt for only Generator (Non RAG Pipelines)")
        prompts = [self.default_prompt.get_string(question=q) for q in queries]
        return prompts

    def custom_prompt(self, queries, references, question_at_beginning=False, question_at_end=True):
        """
        Generate custom prompts including references (retrieval results) for each query.

        :param queries: list of question strings
        :param references: list of document/reference strings corresponding to each question
        :param question_at_beginning: if True, place the question at the beginning of the prompt
        :param question_at_end: if False, remove the question from the end of the prompt
        :return: list of customized prompt strings
        """
        logger.debug(f"Generating custom prompts for {len(queries)} queries with references")

        # Create prompts with document references included
        prompts = [self._custom_prompt.get_string(question=q, retrieval_result=r) for q, r in zip(queries, references)]
        logger.debug(f"Initial custom prompts generated, sample: {prompts[:2]}")

        if question_at_beginning:
            logger.debug("Placing question at the beginning of the prompt")
            prompts = [f'Question: {q}\n' for p, q in zip(prompts, queries)]

        if not question_at_end:
            logger.debug("Removing question at the end of the prompt")
            prompts = [p.replace(f'Question: {q}\n', '') for p, q in zip(prompts, queries)]

        logger.debug(f"Final custom prompts generated, sample: {prompts[:2]}")
        return prompts
