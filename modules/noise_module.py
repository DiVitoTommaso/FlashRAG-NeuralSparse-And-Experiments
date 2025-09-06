from . import *
from .modules_logger import *

class NoiseModule:
    def __call__(self, dataset, results, config):
        """
        Adds random documents from the dataset corpus to each retrieval result to simulate noise.

        :param dataset: dataset object containing the corpus
        :param results: list of current retrieval results (each a list of dicts)
        :param config: configuration dict containing 'random_docs' - number of random docs to add
        :return: new list of retrieval results with noise documents appended
        """
        logger.debug(f"Adding noise: appending {config['random_docs']} random documents to each result")
        new_results = []
        for result in results:
            tmp = result[:]  # copy existing results to avoid modifying original
            for _ in range(config['random_docs']):
                # Choose a random document from the corpus
                row = random.choice(dataset.corpus)
                # Append the random document to the result list
                tmp.append({'id': row['id'], 'contents': row['contents']})
            new_results.append(tmp)
        logger.debug("Noise addition completed")
        return new_results

