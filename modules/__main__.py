from modules import *
if __name__ == '__main__':
    import argparse
    import logging

    logging.basicConfig(level=logging.INFO)  # Set to DEBUG for more verbosity
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Remote FlashRAG Module")
    parser.add_argument(
        "--config",
        default="flashrag/config/full_config.yaml",
        help="Path to the config YAML file for the module (default: flashrag/config/full_config.yaml)"
    )
    parser.add_argument(
        "--run",
        default="generator",
        choices=["retriever", "reranker", "generator"],
        help="Module to host (built with given configuration file)"
    )
    parser.add_argument(
        "--mode",
        default="server",
        choices=["server", "client"],
        help="Run as server or client (for quick testing)"
    )
    parser.add_argument(
        "--addr",
        default="127.0.0.1",
        help="Server bind address or Client test address"
    )
    parser.add_argument(
        "--port",
        default="-",
        help="Server bind port or Client test port"
    )
    args = parser.parse_args()

    configuration = Config(args.config)

    if args.mode == "server":
        # Create module instance for server
        module_instance = None
        if args.run == "generator":
            module_instance = GeneratorModule(configuration)
        elif args.run == "reranker":
            module_instance = RerankModule(configuration)
        elif args.run == "retriever":
            module_instance = RetrieverModule(configuration)

        remote_module = RemoteModule(module_instance, "0.0.0.0")

        logger.info(f"Starting remote module server for {args.run} on 0.0.0.0")
        while True:
            try:
                remote_module()
            except Exception as e:
                logger.error(f"Exception in server loop: {e}")

    elif args.mode == "client":
        # Run client test to connect to remote module and send sample queries
        logger.info(f"Running client mode connecting to {args.ip} for module {args.host}")

        # Create client RemoteModule with module type (class), not instance
        # This triggers client socket connection inside RemoteModule __init__
        client = RemoteModule(
            pipeline_module={
                "generator": GeneratorModule,
                "reranker": RerankModule,
                "retriever": RetrieverModule,
            }[args.run],
            address=args.addr
        )

        # Example queries for testing - adjust or expand this list as needed
        test_queries = [
            "Who is Donald Trump? Give only the answer and do not output any other words. Answer:",
            "What is the capital of France? Only give the answer.  Answer:",
            "Explain quantum computing briefly. Only answer with the explanation.  Answer:"
        ]

        try:
            # Call client RemoteModule with queries, it sends them to server and returns answers and timing
            results, elapsed_time = client(test_queries)
            logger.info(f"Received results: {results}")
            logger.info(f"Elapsed time: {elapsed_time:.3f} seconds")
        except Exception as e:
            logger.error(f"Client failed to get response: {e}")
