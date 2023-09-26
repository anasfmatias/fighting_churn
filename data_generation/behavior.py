import numpy as np
import os
import pandas as pd
from shutil import copyfile
from customer import Customer

def is_pos_def(matrix):
    """
    Check if the given matrix is positive definite.
    
    A matrix is positive definite if all its eigenvalues are positive and it's symmetric.
    :param matrix: np.ndarray
    :return: bool
    """
    rtol = 1e-05
    atol = 1e-08
    return np.all(np.linalg.eigvals(matrix) > 0) and np.allclose(matrix, matrix.T)

class BehaviorModel:
    """
    Base class for modeling customer behavior.
    
    It defines the essential methods that should be implemented by subclasses.
    """

    def generate_customer(self, start_of_month, args):
        """
        Generate a customer instance.
        
        This method is intended to be overridden by subclasses.
        :param start_of_month: Date indicating the start of the month.
        :param args: Additional arguments for the customer generation.
        :return: Customer instance.
        """
        raise NotImplementedError('Sub-classes must define generate_customer!')

    def insert_event_types(self, schema_name, db):
        """
        Ensure the database is set up to record customer events.
        
        This scans the event type table and inserts any events not already present.
        :param schema_name: Name of the database schema.
        :param db: Database connection.
        :return
        """
        for idx, event in enumerate(self.behave_names):
            if db.one(self._event_id_sql(schema_name, event)) is None:
                db.run(f"INSERT into {schema_name}.event_type VALUES ({idx},'{event}');")

    def _event_id_sql(self, schema, event_name):
        """
        Generate SQL to check for the presence of an event in the event type table.
        
        :param schema: Name of the database schema.
        :param event_name: Name of the event.
        :return: SQL query string.
        """
        return f"select event_type_id from {schema}.event_type where event_type_name='{event_name}'"

class NormalBehaviorModel(BehaviorModel):
    """
    Model customer behavior based on a mean and covariance matrix.
    
    This class simulates customer event rates based on Gaussian distributions.
    """

    def __init__(self, name, random_seed=None, version='model'):
        """
        Initialize the model.
        
        Parameters are loaded from a CSV configuration file. The file should define behaviors, means, and covariances.
        :param name: Model name.
        :param random_seed: Seed for random number generation (optional).
        :param version: Version of the model.
        """
        self.name = name
        self.version = version
        model_path = f'{os.path.abspath(os.path.dirname(__file__))}/conf/{name}_{version}.csv'
        model = pd.read_csv(model_path).set_index(['behavior'])
        # Load model parameters from the CSV file
        self.behave_means = model['mean']
        self.behave_maxs = model.get('max')
        self.behave_names = model.index.values
        self.behave_cov = model[self.behave_names]
        
        # Ensure model validity
        if not all([b in model.columns for b in self.behave_names]):
            raise ValueError
        if random_seed:
            np.random.seed(random_seed)

        # Handle non-positive definite covariance matrices
        if not is_pos_def(self.behave_cov) and input("Matrix is not positive semi-definite. Multiply by transpose? (Y to proceed)") in ('y', 'Y'):
            self.behave_cov = np.dot(self.behave_cov, self.behave_cov.T)

        # Backup the original model to another location
        save_path = os.path.join(os.getenv('CHURN_OUT_DIR'), self.name)
        os.makedirs(save_path, exist_ok=True)
        copyfile(model_path, os.path.join(save_path, f'{name}_{version}_simulation_model.csv'))

    def generate_customer(self, start_of_month, args):
        """
        Simulate a customer's event rates.
        
        Event rates are drawn from a multivariate Gaussian distribution.
        :param start_of_month: Date indicating the start of the month.
        :param args: Additional arguments for the customer generation.
        :return: Customer instance with generated event rates.
        """
        rates = np.random.multivariate_normal(mean=self.behave_means, cov=self.behave_cov).clip(min=self.behave_means.min() * 0.01)
        return Customer(pd.DataFrame({'behavior': self.behave_names, 'monthly_rate': rates}), start_of_month, args)

class LogNormalBehaviorModel(NormalBehaviorModel):
    """
    Model customer behavior using a Log-Normal distribution.
    
    This extends the normal behavior model to use a Log-Normal distribution for simulating event rates.
    """

    def __init__(self, name, exp_base, random_seed=None, version=None):
        """
        Initialize the Log-Normal behavior model.
        
        :param name: Model name.
        :param exp_base: Exponential base for the Log-Normal transformation.
        :param random_seed: Seed for random number generation (optional).
        :param version: Version of the model.
        """
        self.exp_base = exp_base
        self.log_fun = lambda x: np.log(x) / np.log(self.exp_base)
        self.exp_fun = lambda x: np.power(self.exp_base, x)
        super().__init__(name, random_seed, version)

    def generate_customer(self, start_of_month, args):
        """
        Simulate a customer's event rates using a Log-Normal distribution.
        :param start_of_month: Date indicating the start of the month.
        :param args: Additional arguments for the customer generation.
        """
        rates = self.exp_fun(np.random.multivariate_normal(mean=self.log_fun(self.behave_means), cov=self.behave_cov))
        rates = np.maximum(rates - 0.667, 0.333)
        if self.behave_maxs is not None:
            rates = rates.clip(max=self.behave_maxs)
        return Customer(pd.DataFrame({'behavior': self.behave_names, 'monthly_rate': rates}), start_of_month, args, channel_name=self.version)