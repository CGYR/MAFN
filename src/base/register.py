# encodingL: utf-8

class Registry():
    """
    The registry that provides name -> object mapping
    To create a registry (e.g. a backbone registry):
        BACKBONE_REGISTRY = Registry('BACKBONE')

    To register an object as decorator:
        @BACKBONE_REGISTRY.register()
        class MyBackbone():
    Or as function call:
        BACKBONE_REGISTRY.register(MyBackbone)
    """

    def __init__(self, name):
        """
        Args:
            name (str): the name of this registry
        """
        self._name = name
        self._obj_map = {}

    def _do_register(self, name, obj, suffix=None):
            if isinstance(suffix, str):
                name = name + '_' + suffix

            assert (name not in self._obj_map), (f"An object named '{name}' was already registered in {self._name}' registry!")
            self._obj_map[name] = obj

    def register(self, obj=None, suffix=None):
        """
        Register the given object under the the name `obj.__name__`.
        Can be used as either a decorator or not.
        """
        if obj is None:
            # used as a decorator
            def deco(func_or_class):
                name = func_or_class.__name__
                self._do_register(name, func_or_class, suffix)
                return func_or_class

            return deco

        # used as a function call
        name = obj.__name__
        self._do_register(name, obj, suffix)

    def get(self, name, suffix=None):
        ret = self._obj_map.get(name)
        if ret is None:
            ret = self._obj_map.get(name + '_' + suffix) if isinstance(suffix,str) else None
            print(f'Name {name} is not found, use name: {name}_{suffix}!')
        if ret is None:
            raise KeyError(f"No object named '{name}' found in '{self._name}' registry!")
        return ret

    def __contains__(self, name):
        return name in self._obj_map

    def __iter__(self):
        return iter(self._obj_map.items()) #返回迭代器

    def keys(self):
        return self._obj_map.keys()

# ModelRegister=Registry('model')
# CollateRegister=Registry("collate")
# GeneratorRegister=Registry('generator')

