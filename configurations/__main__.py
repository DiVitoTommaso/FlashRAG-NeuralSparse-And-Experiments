from flashrag.config import Config
from modules import DatasetModule
from pipeline import PipelineBuilder

def expand_config(config):
    from itertools import product
    options = {
        key: [value] if not isinstance(value, list) and value is not None else value or [None]
        for key, value in config.items()
    }

    keys = list(options.keys())
    all_combinations = [
        dict(zip(keys, values))
        for values in product(*(options[key] for key in keys))
    ]
    return all_combinations

if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Run multiple pipelines")
    parser.add_argument(
        "--config",
        default="flashrag/config/wiki_config.yaml",
        help="Path to the config YAML file (default: flashrag/config/full_config.yaml)"
    )

    parser.add_argument(
        "--run",
        default="configurations/test.yaml",
        help="File with all configurations to test"
    )

    parser.add_argument(
        "--remote",
        default="-",
        help="S (retriever), R (reranker), G (generator). Pass a combination to enable more remote modules. Ex: SRG (To enable all modules as remote)"
    )

    args = parser.parse_args()
    print(args.config)
    configuration = Config(args.config)
    remotes = set(args.remote.lower().split())

    # Load from file
    with open(args.run, 'r') as file:
        params = yaml.safe_load(file)

    combinations = expand_config(params)
    print(f"Total combinations: {len(combinations)}")
    for combo in combinations:
        print(combo)

        configuration['dataset_name'] = combo['name']
        configuration['split'] = combo['split']

        builder = PipelineBuilder(DatasetModule(configuration))

        if combo['retriever']:
            configuration['retrieval_method'] = combo['retriever']
            configuration['retrieval_topk'] = combo['retriever_topk']

            builder.with_retriever(remote='s' in remotes)
            if configuration['retrieval_method'] == 'bm25':
                configuration['index_path'] = 'indexes/bm25/'
                configuration['retrieval_model_path'] = 'retriever/bm25'

            if configuration['retrieval_method'] == 'e5':
                configuration['index_path'] = 'indexes/e5_HNSW32_wiki.index'
                configuration['retrieval_model_path'] = 'retriever/e5-base-v2'

            if configuration['retrieval_method'] == 'splade':
                configuration['index_path'] = 'indexes/splade_flat_wiki_20k.index.seismic'
                configuration['retrieval_model_path'] = 'retriever/splade-v3'

        if combo['reranker']:
            configuration['rerank_model_name'] = combo['reranker'].split('/')[-1]
            configuration['rerank_model_path'] = combo['reranker']
            configuration['rerank_topk'] = combo['reranker_topk']
            builder.with_reranker(remote='r' in remotes)

        if combo['sliding_window']:
            configuration['sliding_window'] = combo['sliding_window']
            builder.with_slide_window()
        if combo['generator']:
            configuration['generator_model'] = combo['generator'] and combo['generator'].split('/')[-1]
            configuration['generator_model_path'] = combo['generator']

            # Da rimuovere
            builder.with_generator(remote='g' in remotes)

        if combo['reorder_strategy']:
            configuration['reorder_strategy'] = combo['reorder_strategy']
            builder.with_documents_reorder()

        if combo['noise_docs']:
            if combo['noise_docs'] == 'fill':
                configuration['random_docs'] = combo['max_docs'] - combo['retriever_topk']
            else:
                configuration['random_docs'] = combo['noise_docs']
            builder.with_noise()

        if combo['refiner']:
            configuration['refiner_name'] = combo['refiner'] and combo['refiner'].split('/')[-1]
            configuration['refiner_model_path'] = combo['refiner']

            if combo['refiner_type'] == 'docs':
                builder.with_prompt_refiner()
            if combo['refiner_type'] == 'answer':
                builder.with_answer_refiner()
            if combo['refiner_type'] == 'both':
                builder.with_prompt_refiner()
                builder.with_answer_refiner()


        try:
            threading = combo['use_threading']
        except Exception as e:
            threading = True


        builder.with_threads(threading)
        builder.with_name(combo['name'])
        builder.with_config(configuration)

        pipeline = builder.build()

        pipeline.run() # Pass prompt strings as arguments to make custom prompt different from default ones
        del pipeline


