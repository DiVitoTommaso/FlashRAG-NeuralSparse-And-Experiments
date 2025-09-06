from . import *
from .modules_logger import *

class RemoteModule:

    @staticmethod
    def send_json(sock, obj):
        """
        Serialize obj to JSON, encode as UTF-8, and send over socket with a 4-byte length prefix.
        """
        data = json.dumps(obj).encode('utf-8')
        length = struct.pack('>I', len(data))  # Pack length as big-endian unsigned int (4 bytes)
        sock.sendall(length + data)
        logger.debug(f"Sent JSON data of length {len(data)} bytes")

    @staticmethod
    def recv_json(sock):
        """
        Receive a JSON object from socket, expecting a 4-byte length prefix followed by JSON data.
        Returns deserialized Python object or None if socket closed.
        """
        # Read 4-byte length prefix
        raw_length = RemoteModule.recvall(sock, 4)
        if not raw_length:
            logger.warning("No length received, socket might be closed")
            return None

        length = struct.unpack('>I', raw_length)[0]
        # Read the JSON data itself
        raw_data = RemoteModule.recvall(sock, length)
        if raw_data is None:
            logger.warning("Incomplete JSON data received")
            return None

        obj = json.loads(raw_data.decode('utf-8'))
        logger.debug(f"Received JSON data of length {length} bytes")
        return obj

    @staticmethod
    def recvall(sock, n):
        """
        Receive exactly n bytes from the socket.
        Returns bytes or None if connection closed before n bytes received.
        """
        data = bytearray()
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                logger.warning("Socket connection closed unexpectedly during recvall")
                return None
            data.extend(packet)
        return data

    def __init__(self, pipeline_module, address):
        """
        Initialize remote module for either client (user) or server (host) mode.

        Args:
            pipeline_module: Either a class type (RetrieverModule, RerankModule, GeneratorModule) for client,
                             or an instance of one of those modules for server.
            address: IP address or hostname for connection or binding.
        """
        self.module = pipeline_module

        # Determine port based on module type
        if type(pipeline_module) == RetrieverModule or pipeline_module == RetrieverModule:
            port = 9498
        elif type(pipeline_module) == RerankModule or pipeline_module == RerankModule:
            port = 9499
        elif type(pipeline_module) == GeneratorModule or pipeline_module == GeneratorModule:
            port = 9500
        else:
            raise Exception('Module not supported for remote connection')

        logger.warning(
            "REMOTE HOSTS WILL USE LOCAL CONFIGURATION FILES! "
            "MAKE SURE ALL THE MACHINES HAVE THE SAME CONFIGURATION FILES AND SAME TOKENIZERS!"
        )

        mode = 'user' if isinstance(pipeline_module, type) else 'host'
        logger.debug(f"Starting {mode.capitalize()} mode. Cluster Address: {address}:{port}")

        if isinstance(pipeline_module, type):
            # Client mode: create socket and connect to remote server
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            logger.debug(f"Connecting to server at {address}:{port}")
            self.socket.connect((address, port))
            logger.info("Connected to remote server")
        else:
            # Server mode: create socket, bind, and listen for clients
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((address, port))
            self.server_socket.listen(5)
            logger.info(f"Server listening on {address}:{port}")

    def __call__(self, *params):
        """
        Handle remote call:
        - If acting as a client (module is a type), send params as JSON to remote server and receive response.
        - If acting as a server (module is an instance), accept client connection, receive JSON,
          process with local module, and send back results.
        """
        if isinstance(self.module, type):
            # Client mode
            with self.socket:
                try:
                    json_obj = {"data": params[0]}
                    if len(params) == 2:
                        json_obj['results'] = params[1]
                    logger.debug(f"[CLIENT] Sending JSON: {json_obj}")
                    RemoteModule.send_json(self.socket, json_obj)

                    response = RemoteModule.recv_json(self.socket)
                    logger.debug(f"[CLIENT] Received JSON: {response}")

                    return response['results'], response['time']
                except Exception as e:
                    logger.error(f"[CLIENT] Error during remote call: {e}")
                    raise
        else:
            # Server mode: wait for a client connection and handle request
            self.socket, client_address = self.server_socket.accept()
            logger.info(f"[SERVER] Accepted connection from {client_address}")
            with self.socket:
                try:
                    json_obj = RemoteModule.recv_json(self.socket)
                    logger.debug(f"[SERVER] Received JSON: {json_obj}")

                    # Check if 'results' field present, pass accordingly
                    if 'results' in json_obj:
                        results, time_taken = self.module(json_obj['data'], json_obj['results'])
                    else:
                        results, time_taken = self.module(json_obj['data'])

                    response = {"status": "ok", "results": results, "time": time_taken}
                    RemoteModule.send_json(self.socket, response)
                    logger.debug(f"[SERVER] Sent JSON: {response}")
                except Exception as e:
                    logger.error(f"[SERVER] Error during request handling: {e}")
                    # Optional: send error response to client or close connection
