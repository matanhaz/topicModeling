# from .Singelton import Singleton


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton,
                                        cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Experiment_Data(metaclass=Singleton):
    def __init__(self):
        self.PRIORS = []
        self.BUGS = []
        self.POOL = {}
        self.ESTIMATED_POOL = {}
        self.COMPONENTS_NAMES = {}
        self.REVERSED_COMPONENTS_NAMES = {}
        self.experiment_type = None
        self.clear()

    def clear(self):
        self.PRIORS = []
        self.BUGS = []
        self.POOL = {}
        self.COMPONENTS_NAMES = {}
        self.REVERSED_COMPONENTS_NAMES = {}

    def set_values(self,
                   priors_arg,
                   bugs_arg,
                   pool_arg,
                   components_arg,
                   extimated_pool_arg=None,
                   experiment_type=None,
                   **kwargs):
        self.clear()
        self.PRIORS = priors_arg
        self.BUGS = list(map(lambda x: x.lower(), bugs_arg))
        self.POOL = pool_arg
        self.COMPONENTS_NAMES = components_arg
        self.REVERSED_COMPONENTS_NAMES = dict(
            map(lambda x: tuple(reversed(x)), self.COMPONENTS_NAMES.items()))
        self.ESTIMATED_POOL = extimated_pool_arg
        self.experiment_type = experiment_type
        list(map(lambda attr: setattr(self, attr, kwargs[attr]), kwargs))
        tmp = list(map(self.get_component_id, self.BUGS))
        assert None not in tmp

    def get_experiment_type(self):
        return self.experiment_type

    def get_component_id(self, component_name):
        return self.REVERSED_COMPONENTS_NAMES.get(component_name, None)

    def get_named_bugs(self):
        return list(map(self.COMPONENTS_NAMES.get, self.BUGS))

    def get_id_bugs(self):
        ret_value = list(map(self.get_component_id, self.BUGS))
        return ret_value
