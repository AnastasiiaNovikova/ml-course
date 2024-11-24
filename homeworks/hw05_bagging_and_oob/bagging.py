import numpy as np

class SimplifiedBaggingRegressor:
    def __init__(self, num_bags, oob=False):
        self.num_bags = num_bags
        self.oob = oob

    def _generate_splits(self, data: np.ndarray):
        '''
        Generate indices for every bag and store in self.indices_list list
        '''
        self.indices_list = []
        data_length = len(data)
        for bag in range(self.num_bags):
            # Your Code Here
            bag_indices = np.random.choice(data_length, data_length, replace=True)
            self.indices_list.append(bag_indices)

    def fit(self, model_constructor, data, target):
        '''
        Fit model on every bag.
        Model constructor with no parameters (and with no ()) is passed to this function.

        example:

        bagging_regressor = SimplifiedBaggingRegressor(num_bags=10, oob=True)
        bagging_regressor.fit(LinearRegression, X, y)
        '''
        self.data = None
        self.target = None
        self._generate_splits(data)
        assert len(set(list(map(len, self.indices_list)))) == 1, 'All bags should be of the same length!'
        assert list(map(len, self.indices_list))[0] == len(
            data), 'All bags should contain `len(data)` number of elements!'
        self.models_list = []

        for bag_indices in self.indices_list:
            model = model_constructor()
            data_bag, target_bag = data[bag_indices], target[bag_indices]  # Your Code Here
            self.models_list.append(model.fit(data_bag, target_bag))  # store fitted models here

        if self.oob:
            self.data = data
            self.target = target

    def predict(self, data):
        '''
        Get average prediction for every object from passed dataset
        '''
        # Your code here
        predictions = np.zeros(len(data))
        for model in self.models_list:
            predictions += model.predict(data)

        return predictions / self.num_bags

    def _get_oob_predictions_from_every_model(self):
        '''
        Generates list of lists, where list i contains predictions for self.data[i] object
        from all models, which have not seen this object during training phase
        '''
        list_of_predictions_lists = [[] for _ in range(len(self.data))]
        # Your Code Here
        for model, bag_indices in zip(self.models_list, self.indices_list):
            oob_indices = set(range(len(self.data))) - set(bag_indices)
            for idx in oob_indices:
                prediction = model.predict(self.data[idx].reshape(1, -1))[0]
                list_of_predictions_lists[idx].append(prediction)

        self.list_of_predictions_lists = np.array(list_of_predictions_lists, dtype=object)

    def _get_averaged_oob_predictions(self):
        '''
        Compute average prediction for every object from training set.
        If object has been used in all bags on training phase, return None instead of prediction
        '''
        self._get_oob_predictions_from_every_model()
        # Your Code Here
        self.oob_predictions = np.array([
            np.mean(predictions) if predictions else None
            for predictions in self.list_of_predictions_lists
        ])

    def OOB_score(self):
        '''
        Compute mean square error for all objects, which have at least one prediction
        '''
        self._get_averaged_oob_predictions()
        # Your Code Here
        valid_indices = [i for i, preds in enumerate(self.oob_predictions) if preds is not None]
        if not valid_indices:
            return None
        return np.mean((self.oob_predictions[valid_indices] - self.target[valid_indices]) ** 2)