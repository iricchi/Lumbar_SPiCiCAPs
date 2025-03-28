from .utils import ar_mls
from ..data import GroupData, Results


class VAR:
    def __init__(self, order=1):
        self._order = order

    def fit(self, data):
        ''' Fits VAR model to Data.
        Args:
            data (GroupData or Results): Data to fit to.
        '''
        if isinstance(data, GroupData):
            comps = data.get_results()
        else:
            comps = data

        # Get timecourses
        timecourses = comps.time_courses()

        # Put in list of tcs format
        input_tcs = [timecourses[:, i, :] for i in range(timecourses.shape[1])]
        self.fit_from_tcs(input_tcs)

    def fit_from_tcs(self, input_tcs):
        ''' Fits VAR model to timecourses.
        Args:
            input_tcs (list of np.ndarray): List of timecourses for every subject.
        '''
        # Get VAR parameters
        _, _, self._params, self._residuals = ar_mls(input_tcs, self._order)
    
    def get_params(self):
        return self._params
    
    def get_residuals(self):
        return self._residuals
