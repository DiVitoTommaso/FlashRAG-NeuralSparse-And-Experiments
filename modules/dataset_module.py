from . import *
from .modules_logger import *

class DatasetModule:
    def __init__(self, config):
        """
        Load the dataset based on the provided config.
        Also cleans up any empty gold answers by replacing them with "[No answer]".
        """
        logger.debug("Loading dataset")
        self._dataset = get_dataset(config)
        self._dataset = self._dataset[config['split'][0]]  # Select the specified split (e.g., train/test)

        # Replace any empty gold answers with a placeholder string
        for i in range(len(self._dataset.golden_answers)):
            for j in range(len(self._dataset.golden_answers[i])):
                if not self._dataset.golden_answers[i][j]:
                    self._dataset.golden_answers[i][j] = "[No answer]"
                    logger.debug(f"Missing answer at question: {i}. Replacing with [No answer]")

        self.config = config
        self._corpus = None  # Lazy-loaded corpus data
        self._queries = None  # Lazy-loaded questions
        self._answers = None  # Lazy-loaded gold answers

    @property
    def all(self):
        """
        Return the full underlying dataset object.
        """
        return self._dataset

    @property
    def corpus(self):
        """
        Load and cache the corpus from disk on first access.
        """
        if self._corpus is None:
            self._corpus = load_corpus(self.config['corpus_path'])
            logger.debug(f"Loaded {len(self._corpus)} corpus lines")
        return self._corpus

    @property
    def queries(self):
        """
        Return the list of questions in the dataset, loading on first access.
        """
        if self._queries is None:
            self._queries = self._dataset.question
            logger.debug(f"Loaded {len(self._queries)} questions")
        return self._queries

    @property
    def answers(self):
        """
        Return the list of gold answers, loading on first access.
        Note: there is a bug here (calls self.answers recursively) — fix below.
        """
        if self._answers is None:  # <-- fixed condition to check _answers instead of answers
            self._answers = self._dataset.golden_answers
            logger.debug(f"Loaded {len(self._answers)} gold answers")
        return self._answers

    @property
    def predictions(self):
        """
        Get the current list of model predictions stored in dataset.
        """
        return self.all.pred

    @predictions.setter
    def predictions(self, predictions):
        """
        Set or update the predictions in the underlying dataset.
        """
        self.all.update_output('pred', predictions)

    @property
    def retrieval(self):
        """
        Getter for the retrieval results.
        Returns the current retrieval results stored in the dataset's 'all' attribute.
        """
        return self.all.last_retrieval_results

    @retrieval.setter
    def retrieval(self, retrieval):
        """
        Setter for the retrieval results.
        Updates the dataset's 'all' attribute by setting the 'retrieval_results' key to the given retrieval value.
        """
        self.all.update_output('last_retrieval_results', retrieval)

    def __getitem__(self, item):
        """
        Enable indexing syntax (e.g., obj[key]) to access underlying dataset items.
        """
        return self.all[item]

    def __setitem__(self, key, value):
        """
        Enable setting items (e.g., obj[key] = value) to update dataset outputs.
        """
        self.all.update_output(key, value)