from . import *
from .modules_logger import *

class MPIRemoteModule:

    def __init__(self, pipeline_module):
        from mpi4py import MPI
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        """
        Initialize remote module for either client (user) or server (host) mode.

        Args:
            pipeline_module: Either a class type (RetrieverModule, RerankModule, GeneratorModule) for client,
                             or an instance of one of those modules for server.
        """
        self.module = pipeline_module

        if isinstance(pipeline_module, type):
            # Client mode
            self.mode = 'user'
        else:
            # Server mode
            self.mode = 'host'

        logger.warning(
            "REMOTE HOSTS WILL USE LOCAL CONFIGURATION FILES! "
            "MAKE SURE ALL THE MACHINES HAVE THE SAME CONFIGURATION FILES AND SAME TOKENIZERS!"
        )
        logger.debug(f"Starting {self.mode.capitalize()} mode. MPI Rank: {self.rank}")

    def __call__(self, *params):
        """
        Handle remote call:
        - If acting as a client, send params as JSON to remote server and receive response.
        - If acting as a server, receive JSON, process with local module, and send back results.
        """
        if self.mode == 'user':
            json_obj = {"data": params[0]}
            if len(params) == 2:
                json_obj['results'] = params[1]
            logger.debug(f"[CLIENT] Sending JSON: {json_obj}")

            self.comm.send(json.dumps(json_obj), dest=1, tag=0)
            response_json = self.comm.recv(source=1, tag=1)
            response = json.loads(response_json)
            logger.debug(f"[CLIENT] Received JSON: {response}")
            return response['results'], response['time']
        else:
            # Server loop should be handled externally by calling serve_forever()
            raise RuntimeError("Cannot call module directly in host mode. Use `serve_forever`.")

    def serve_forever(self):
        """
        Run the server loop on rank 1+.
        """
        if self.mode != 'host':
            raise RuntimeError("Only host mode can run the server loop.")

        logger.info(f"[SERVER] Rank {self.rank} ready to receive tasks.")
        while True:
            try:
                request_json = self.comm.recv(source=0, tag=0)
                json_obj = json.loads(request_json)
                logger.debug(f"[SERVER] Received JSON: {json_obj}")

                start_time = time.time()
                if 'results' in json_obj:
                    results, _ = self.module(json_obj['data'], json_obj['results'])
                else:
                    results, _ = self.module(json_obj['data'])
                time_taken = time.time() - start_time

                response = {"status": "ok", "results": results, "time": time_taken}
                self.comm.send(json.dumps(response), dest=0, tag=1)
                logger.debug(f"[SERVER] Sent JSON: {response}")
            except Exception as e:
                logger.error(f"[SERVER] Error during request handling: {e}")
                break
