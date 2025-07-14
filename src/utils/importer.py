import importlib
from typing import Optional


def load_class(path: str, default_pkg: Optional[str] = None):
    """
    Dynamically load a class, given either:
      - a full import path: "mypkg.mymodule.MyClass"
      - a bare class name: "MyClass", in which case
        default_pkg must be provided (e.g. "src.models").
    """
    if "." in path:
        module_name, class_name = path.rsplit(".", 1)
    else:
        if default_pkg is None:
            raise ValueError("Must provide default_pkg for bare class names")
        module_name, class_name = default_pkg, path

    module = importlib.import_module(module_name)
    try:
        return getattr(module, class_name)
    except AttributeError:
        raise ImportError(f"Module '{module_name}' has no attribute '{class_name}'")
