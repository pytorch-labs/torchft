import socket
from http.server import ThreadingHTTPServer


class _IPv6HTTPServer(ThreadingHTTPServer):
    address_family: socket.AddressFamily = socket.AF_INET6
    request_queue_size: int = 1024
