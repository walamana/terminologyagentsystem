from importlib import import_module

__lazy_modules: dict = {}

def lazy_module(module: str):
    global __lazy_modules
    if module not in __lazy_modules:
        __lazy_modules[module] = import_module(module)
    return __lazy_modules[module]