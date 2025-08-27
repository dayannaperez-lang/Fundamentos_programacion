from .io_multiprocessing import  multiprocessing_method
from .io_threading import threading_method
from .io_synchronous import basic_request_method
from .io_asyncio import asyncio_method

__all__ = ['multiprocessing_method',
            'threading_method',
            'basic_request_method', 
            'asyncio_method']


