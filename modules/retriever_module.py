import json

from . import *
from .modules_logger import *

class RetrieverModule:
    def __init__(self, config, multi_thread=True):
        """
        Initialize the RetrieverModule with a retriever instance based on the provided config.

        :param config: configuration parameters needed to instantiate the retriever
        """
        self.config = config
        self._retriever = get_retriever(config)
        self.multi_thread = multi_thread
        logger.debug("Retriever initialized")

    def __call__(self, queries, return_score=False):
        """
        Perform batch retrieval for a list of queries.

        :param queries: list of query strings to retrieve documents for
        :return: a tuple (results, elapsed_time) where results is the retrieval output
                 and elapsed_time is the time taken to perform retrieval in seconds
        """
        if self.config['use_retrieval_cache']:
            if os.path.exists('retrieval_cache.jsonl') and os.path.getsize('retrieval_cache.jsonl') > 0:
                with open('retrieval_cache.jsonl', 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                            if (
                                    entry['method'] == self.config['retrieval_method'] and
                                    entry['index'] == self.config['index_path'] and
                                    entry['corpus'] == self.config['corpus_path'] and
                                    entry['topk'] == self.config['retrieval_topk'] and
                                    entry['pooling'] == self.config['retrieval_pooling_method'] and
                                    entry['dataset'] == self.config['dataset_name'] and
                                    entry['split'] == self.config['split']
                            ):
                                return entry['docs']
                        except (json.JSONDecodeError, KeyError):
                            print("Internal error")
                            continue  # Skip malformed or incomplete entries


        logger.debug("Retrieval phase started")
        # Perform batch search using the retriever instance
        if self.multi_thread:
            results, time = self._retriever.batch_search(queries, return_score=return_score)
            logger.debug(f"Retrieval phase ended")
        else:
            results = []
            time_total = []
            for query in queries:
                tmp = self._retriever.batch_search(query, return_score=return_score)
                results.extend(tmp[0])
                time_total.append(tmp[1])

            final_time = {}
            for d in time_total:
                for key, value in d.items():
                    final_time[key] = final_time.get(key, 0) + value

            logger.debug(f"Retrieval phase ended")
            time = final_time

        if self.config['save_retrieval_cache']:
            with open('retrieval_cache.jsonl', 'a') as f:
                tmp = {'dataset': self.config['dataset_name'], 'split': self.config['split'],
                       'method': self.config['retrieval_method'], 'index': self.config['index_path'],
                       'corpus': self.config['corpus_path'], 'topk': self.config['retrieval_topk'],
                       'pooling': self.config['retrieval_pooling_method'], 'docs': results}
                f.write(json.dumps(tmp) + '\n')
        return results, time