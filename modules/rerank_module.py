from . import *
from .modules_logger import *

class RerankModule:
    def __init__(self, config):
        """
        Initialize the RerankModule with a reranker instance and configuration.

        :param config: configuration dict including 'reorder_strategy'
        """
        self._reranker = get_reranker(config)
        self.config = config
        logger.debug("Reranker initialized")

    def __call__(self, queries, results):
        """
        Optionally rerank the retrieval results unless the reorder strategy is random.

        :param queries: list of query strings
        :param results: list of retrieval results (lists of dicts with 'contents')
        :return: tuple (reranked_results, elapsed_time_seconds)
        """
        if self.config['reorder_strategy'] != 'random':
            logger.debug("Rerank phase started")
            start = time.time()

            # Extract the 'contents' field from each retrieved document per query
            contents_lists = [[r['contents'] for r in result] for result in results]

            # Perform reranking
            reranked_results, scores = self._reranker.rerank(queries, contents_lists)

            # Wrap the reranked texts back into dicts with 'contents' keys
            reranked_results = [[{'contents': r} for r in result] for result in reranked_results]

            end = time.time()
            elapsed = end - start
            logger.debug(f"Rerank phase ended, took {elapsed:.3f} seconds")
            return reranked_results, elapsed
        else:
            logger.debug("Rerank phase skipped due to '[random]' reorder strategy")
            return results, 0.0

