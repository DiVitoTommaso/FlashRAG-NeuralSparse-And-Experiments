from . import *
from .modules_logger import *
class ReorderModule:

    @staticmethod
    def alternate_start_then_reverse_rest(lst):
        """
        Reorder the list by taking every other element starting from index 0,
        then appending the reversed remaining elements.

        For example: [1, 2, 3, 4] -> [1, 3, 4, 2]
        """
        first_part = lst[::2]  # Elements at even indices
        second_part = lst[1::2]  # Elements at odd indices
        reordered = first_part + second_part[::-1]  # Append reversed odd-index elements
        return reordered

    def __call__(self, config, results):
        """
        Reorder the retrieval results based on the 'reorder_strategy' in config.

        Supported strategies:
        - 'alternated': apply the alternate_start_then_reverse_rest strategy
        - 'reversed': reverse the list
        - 'random': shuffle the list randomly

        :param config: dict containing 'reorder_strategy'
        :param results: list of lists to reorder
        :return: reordered list of lists
        """
        strategy = config['reorder_strategy']
        logger.debug(f"Reorder phase started with strategy: {strategy}")

        if strategy == 'alternated':
            results = [self.alternate_start_then_reverse_rest(result) for result in results]

        elif strategy == 'reversed':
            results = [list(reversed(result)) for result in results]

        elif strategy == 'random':
            for result in results:
                random.shuffle(result)  # in-place shuffle
        else:
            logger.debug("No reordering strategy applied or unknown strategy")

        logger.debug("Reorder phase ended")
        # Return a shallow copy of the reordered results
        return results
