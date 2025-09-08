"""
This is the implementation of the FlashRAG pipeline.
A builder is also included to simplify pipeline creation and usage.

---------------------------------------------------------------------------------
To run a custom pipeline:
    python pipeline.py --config_path <path> --name <name>
---------------------------------------------------------------------------------

This implementation follows the FlashRAG specifications. You can use local models
or specify a Hugging Face repository name — both options behave the same way.

To distribute the pipeline across multiple nodes in your cluster, set the appropriate
address in each node's configuration file and pass `remote=True` to the builder
functions for supported modules.

Note: the master node (i.e., the node running `pipeline.py`) may still perform some
operations involving models hosted on remote machines. To avoid compatibility issues,
PLEASE ENSURE THAT ALL NODES USE THE SAME CONFIGURATION FILE.
A Slurm cluster with a shared disk is highly recommended.

IMPORTANT:
For prompt generation, the master node uses the tokenizer of the model specified in
its configuration file. If the model defined on the remote node differs from the one
on the master node, compatibility issues may arise, and the model might fail or
produce incorrect results. Ensure the models of all configuration files are the same.
"""

from flashrag.pipeline import *

from modules import *
import logging

from modules.refiner_module import RefinerModule

logger = logging.getLogger(__name__)


class PipelineBuilder:
    def __init__(self, dataset_module=None):
        # Initialize configuration dictionary with default values for each component/feature
        self.config = {}
        self.config.setdefault("name", "Test")  # Name of the pipeline (default "Test")
        self.config.setdefault("retriever", False)  # Whether to include retriever module
        self.config.setdefault("noise", False)  # Whether to include noise handling
        self.config.setdefault("prompt_refiner", False)  # Whether to include documents refiner
        self.config.setdefault("input_strategy", False)  # Whether to reorder inputs/documents
        self.config.setdefault("answer_refiner", False)  # Whether to include answer refiner
        self.config.setdefault("evaluation", True)  # Whether to include evaluation (default True)
        self.config.setdefault("reranker", False)  # Whether to include reranker module
        self.config.setdefault("slide_window", False)  # Whether to use sliding window strategy
        self.config.setdefault("prompt", False)  # Whether to include prompt handling
        self.config.setdefault("use_threads", True)
        # Store optional dataset module (can be None)
        self.config['dataset'] = dataset_module

    def with_retriever(self, remote=False):
        # Enable retriever module, optionally remote
        self.config['retriever'] = True
        self.config['remote_retriever'] = remote
        return self

    def with_noise(self):
        # Enable noise handling module
        self.config['noise'] = True
        return self

    def with_slide_window(self):
        # Enable sliding window strategy for input processing
        self.config['slide_window'] = True
        return self

    def with_reranker(self, remote=False):
        # Enable reranker module, optionally remote
        self.config['reranker'] = True
        self.config['remote_reranker'] = remote
        return self

    def with_prompt_refiner(self):
        # Enable documents refiner module
        self.config['prompt_refiner'] = True
        return self

    def with_answer_refiner(self):
        # Enable answer refiner module
        self.config['answer_refiner'] = True
        return self

    def with_evaluation(self):
        # Enable evaluation (default is True)
        self.config['evaluation'] = True
        return self

    def with_documents_reorder(self):
        # Enable document/input reorder strategy
        self.config['input_strategy'] = True
        return self

    def with_generator(self, remote=False):
        # Enable generator module, optionally remote
        self.config['generator'] = True
        self.config['remote_generator'] = remote
        return self

    def with_config(self, config):
        # Provide the configuration object/dictionary (mandatory before build)
        self.config['config'] = config
        return self

    def with_name(self, name):
        # Set a custom name for the pipeline instance
        self.config['name'] = name
        return self

    def with_threads(self, use_threads):
        self.config['use_threads'] = use_threads
        return self

    def build(self):
        # Finalize and construct the Pipeline instance with configured modules
        if self.config["config"] is None:
            raise RuntimeError("No configuration file specified")

        # Create Pipeline instance with name, config, and optional dataset
        pipe = Pipeline(self.config["name"], self.config["config"], self.config["dataset"], multi_thread=self.config['use_threads'])
        # Setup the pipeline modules according to the config flags
        pipe.set_modules(**self.config)
        return pipe


def normal_evaluation(dataset_module, evaluation, pipeline_results):
    pipeline_results['standard_predictions'] = {}
    pipeline_results['standard_predictions'] = pipeline_results[
                                                   'standard_predictions'] | evaluation.eval_advanced()
    pipeline_results['standard_predictions'] = pipeline_results[
                                                   'standard_predictions'] | evaluation.eval_standard()
    dataset_module['original_pred'] = dataset_module.predictions


def fixed_evaluation(dataset_module, evaluation, pipeline_results):
    fixed_predictions = []
    for pred in dataset_module.predictions:
        pred = (pred
                .split('\n', 1)[0]  # usually appear first
                .split("Question:", 1)[0]  # usually appear second
                .split("Final Answer:", 1)[0]  # usually appear third
                .replace("`", "")
                .strip()
                )
        if pred:
            fixed_predictions.append(pred)
        else:
            fixed_predictions.append("No answer found")
    dataset_module['pred'] = fixed_predictions
    pipeline_results['fixed_predictions'] = {}
    pipeline_results['fixed_predictions'] = pipeline_results['fixed_predictions'] | evaluation.eval_advanced()
    pipeline_results['fixed_predictions'] = pipeline_results['fixed_predictions'] | evaluation.eval_standard()
    dataset_module['fixed_pred'] = dataset_module.all.original_pred


class Pipeline(BasicPipeline):
    def __init__(self, name, config, dataset_module=None, multi_thread=True):
        # Initialize the base pipeline class
        super().__init__(config, None)

        if config['openai_setting']['api_key']:
            import os
            os.environ["OPENAI_API_KEY"] = config['openai_setting']['api_key']

        self.name = name  # Store pipeline name
        self.config = config  # Store configuration dictionary/object
        self.modules = {}  # Initialize empty dict to hold modules
        self.multi_thread = multi_thread
        # Initialize dataset module:
        # Use provided dataset_module if it's a DatasetModule instance,
        # otherwise create a new DatasetModule using the config
        if dataset_module and isinstance(dataset_module, DatasetModule):
            self.dataset = dataset_module
            logger.info(f"Using provided DatasetModule instance for pipeline '{self.name}'.")
        else:
            self.dataset = DatasetModule(config)
            logger.info(f"Created new DatasetModule instance for pipeline '{self.name}'.")

    def set_modules(self, **modules):
        # Assign the modules dictionary with provided keyword args
        self.modules = modules

        # Log the list of module keys that were set
        module_names = ', '.join(modules.keys())
        logger.info(f"Modules set for pipeline '{self.name}': {module_names}")

    def run(self, prompt_system=None, prompt_user=None):
        logger.info(f"Starting pipeline run for '{self.name}'")

        pipeline_results = {}

        dataset = self.dataset
        config = self.config

        # Initialize empty lists for various result stages in dataset,
        # one list per query (to store results at each step)
        tmp = [[] for _ in range(len(dataset.queries))]
        for key in ["initial_retrieval_results", "last_retrieval_results", "noise_results",
                    "rerank_results", "reorder_results", "prompt", "pred"]:
            dataset[key] = tmp
            logger.debug(f"Initialized dataset key '{key}' with empty lists for each query")

        # Check if no retriever module is enabled (only generation)
        if not self.modules["retriever"]:
            logger.info("No retriever module enabled, running generation only")

            prompt = PromptModule(self.config, prompt_system, prompt_user)
            input_prompts = prompt.generator_only_prompt(dataset.queries)
            logger.debug(f"Generated {len(input_prompts)} input prompts")

            # Initialize generator module; use remote or local based on config
            generator = self.load_generator(self.config)

            # Run generation phase and measure elapsed time
            pred_answer_list, time = generator(input_prompts)
            logger.info(f"Generation phase completed in {time:.2f} seconds")

            # Clean up generator if needed (nothing happens if generator is a remote module
            del generator
            gpu_release()

            pipeline_results["generation_time"] = time
            dataset["pred"] = pred_answer_list
            self.save_results(dataset, pipeline_results)
            return

        # RETRIEVAL PHASE => Perform batch search using the configured retriever module
        search_engine = self.load_retriever(config)

        # Execute the retrieval over the dataset queries and measure time taken
        results, time = search_engine(dataset.queries, return_score=False)
        print(time)
        logger.info(f"Retrieval required {time['encode_time']:.3f} seconds for encoding and"
                    f" {time['search_time']:.3f} seconds for search. Returned {len(results)} results")

        # Token rates Retrieved Docs - Real answer
        tokens_rates = []
        for result, answers in zip(results, dataset.answers):
            all_doc_tokens = set()
            for doc in result:
                all_doc_tokens.update(doc['contents'].lower().split())

            # Calcola la copertura token per ogni risposta
            max_rate = 0.0
            for ans in answers:  # answers è una lista di possibili risposte per la query
                ans_tokens = set(ans.lower().split())
                if len(ans_tokens) == 0:
                    continue
                match_count = sum(1 for token in ans_tokens if token in all_doc_tokens)
                rate = match_count / len(ans_tokens)
                max_rate = max(max_rate, rate)

            tokens_rates.append(max_rate)

        pipeline_results["tokens_rates"] = tokens_rates
        # Clean up search engine resources if needed
        del search_engine
        gpu_release()

        # Store timing and results in the pipeline results and dataset
        pipeline_results["retrieval_time"] = time
        dataset["initial_retrieval_results"] = results[:]
        dataset["last_retrieval_results"] = results[:]

        # If noise addition is enabled, add random documents to results to increase diversity
        if self.modules["noise"]:
            logger.info("Adding noise documents to retrieval results")
            noise = NoiseModule()
            results = noise(dataset, results, config)
            dataset["noise_results"] = results[:]
            dataset["last_retrieval_results"] = results[:]
            logger.info(f"Noise phase added {config['random_docs']} random documents per query")

        # RERANK PHASE => Reorder the retrieved documents based on reranker module
        if self.modules["reranker"]:
            logger.info("Starting rerank phase")
            rerank = self.load_reranker(config)
            results, time = rerank(dataset.queries, results)
            logger.info(f"Rerank phase completed in {time:.3f} seconds")

            pipeline_results["rerank_time"] = time
            dataset["rerank_results"] = results[:]
            dataset["last_retrieval_results"] = results[:]

            # Clean up reranker resources and clear memory if necessary
            del rerank
            gpu_release()

        # GENERATOR INPUTS => Rearrange input documents if input_strategy enabled
        if self.modules['input_strategy']:
            logger.info("Starting document reorder phase")
            reorder = ReorderModule()
            results = reorder(config, results)
            logger.info("Documents reordered using input_strategy")

            dataset["reorder_results"] = results[:]
            dataset["last_retrieval_results"] = results[:]

        # PROMPT GENERATION => Create base prompt by combining documents and question if prompt module enabled
        if self.modules['slide_window']:
            window = SlideWindowModule(self.config)

            next = window(dataset.corpus, dataset.retrieval)

            all_predictions = []
            if next:
                all_predictions = [[] for _ in range(len(dataset.queries))]

            while next:
                template = PromptModule(config, prompt_system, prompt_user)
                input_prompts = template.custom_prompt(dataset.queries, next)
                logger.debug(f"Generated {len(input_prompts)} input prompts")

                next = window(dataset.corpus, dataset.retrieval)

                # REFINER => Optionally refine prompts (e.g., cleaner, formatted)
                if self.modules['prompt_refiner']:
                    input_prompts = self.input_refine(dataset, input_prompts, pipeline_results)

                dataset["prompt"] = input_prompts

                # GENERATION PHASE => Process each prompt to generate final answers
                generator = self.load_generator(config)
                predictions, time = generator(input_prompts)

                del generator
                gpu_release()

                if len(predictions) != len(all_predictions):
                    raise ValueError("Number of predictions does not match the number of queries. Check generator output")

                for k in range(len(predictions)):
                    all_predictions[k].append(predictions[k])

            evaluation = EvaluationModule(self)
            predictions = evaluation.slide_merge(all_predictions, dataset.queries, dataset.answers, eval_on='em')
        else:

            template = PromptModule(config, prompt_system, prompt_user)
            input_prompts = template.custom_prompt(dataset.queries, results)
            logger.debug(f"Generated {len(input_prompts)} input prompts")

            # REFINER => Optionally refine prompts (e.g., cleaner, formatted)
            if self.modules['prompt_refiner']:
                input_prompts = self.input_refine(dataset, input_prompts, pipeline_results)

            dataset["prompt"] = input_prompts

            logger.info("Starting generation phase")

            # GENERATION PHASE => Process each prompt to generate final answers
            generator = self.load_generator(config)
            predictions, time = generator(input_prompts)

            del generator
            gpu_release()
            logger.info(f"Generation phase completed in {time:.3f} seconds")

        pipeline_results["generation_time"] = time

        # REFINER => Refine answer TO DO IMPLEMENT ANSWER REFINER
        if self.modules['answer_refiner']:
            self.answer_refine(config, dataset, pipeline_results, predictions)
        else:
            dataset["pred"] = predictions

        self.save_results(dataset, pipeline_results)
        gpu_release()

    def answer_refine(self, config, dataset, pipeline_results, predictions):
        refiner = RefinerModule(config)
        predictions, time = refiner(predictions)
        dataset["pred"] = predictions
        pipeline_results["answer_refine_time"] = time
        del refiner
        gpu_release()

    def load_reranker(self, config):
        if self.modules['remote_reranker']:
            logger.info("Using remote reranker module")
            rerank = RemoteModule(RerankModule, self.config['remote_reranker_address'])
        else:
            logger.info("Using local reranker module")
            rerank = RerankModule(config)
        return rerank

    def load_retriever(self, config):
        if self.modules['remote_retriever']:
            logger.info("Using remote retriever module for batch search")
            search_engine = RemoteModule(RetrieverModule, self.config['remote_retriever_address'])
        else:
            logger.info("Using local retriever module for batch search")
            search_engine = RetrieverModule(config, self.multi_thread)
        return search_engine

    def load_generator(self, config):
        if self.modules['remote_generator']:
            logger.info("Using remote generator module")
            generator = RemoteModule(GeneratorModule, self.config['remote_generator_address'])
        else:
            logger.info("Using local generator module")
            generator = GeneratorModule(config)
        return generator

    def input_refine(self, dataset, input_prompts, pipeline_results):
        logger.info("Starting documents refiner phase")
        refiner = get_refiner(self.config)
        input_prompts, time = refiner(input_prompts)
        logger.info(f"Prompts refined in {time:.3f} seconds")
        dataset["prompt"] = input_prompts
        pipeline_results["docs_refine_time"] = time
        del refiner
        gpu_release()
        return input_prompts

    def save_results(self, dataset_module, pipeline_results):
        def has(obj, attr):
            try:
                return hasattr(obj, attr)
            except KeyError:
                return False

        if self.modules["evaluation"]:
            evaluation = EvaluationModule(self)

            normal_evaluation(dataset_module, evaluation, pipeline_results)
            # self.fixed_evaluation(dataset_module, evaluation, pipeline_results)

            pipeline_results['output'] = []
            for item in dataset_module.all:
                if self.config['save_intermediate_data']:
                    pipeline_results['output'].append(
                        {'question': item.question, 'answers': item.golden_answers,
                         'generated_original': item.original_pred,
                         'initial_retrieval_results': item.initial_retrieval_results,
                         'last_retrieval_results': item.last_retrieval_results, 'rerank_results': item.rerank_results,
                         'generated_fixed': item.fixed_pred if has(item, 'fixed_pred') else ''
                         }
                    )
                else:
                    pipeline_results['output'].append(
                        {'question': item.question, 'answers': item.golden_answers,
                         'generated_original': item.original_pred, 'generated_fixed':
                             item.fixed_pred if has(item, 'fixed_pred') else ''
                         }
                    )

        config = eval(repr(self.config))
        pipeline_results = pipeline_results | config
        # SAVE TO FILE => Append to file full results: config + metrics
        with open(self.name, 'a') as f:
            f.write(str(pipeline_results) + '\n')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run RAG pipeline")
    parser.add_argument(
        "--config_path",
        default="flashrag/config/wiki_config.yaml",
        help="Path to the config YAML file (default: flashrag/config/full_config.yaml)"
    )

    parser.add_argument(
        "--name",
        default="Test-Pipeline",
        help="Name of pipeline (default: Test)"
    )

    args = parser.parse_args()

    configuration = Config(args.config_path)
    dataset = DatasetModule(configuration)

    pipeline = (PipelineBuilder(dataset)
                .with_name("Test-Pipeline")
                .with_config(configuration)
                .with_retriever(remote=False)
                .with_generator(remote=False)
                # .with_noise()
                # .with_reranker(remote=False)
                # .with_documents_refiner()
                #.with_documents_reorder()
                # .with_answer_refiner()
                .with_evaluation()
                .build())
    pipeline.run()
