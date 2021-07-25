class Record(object):
    ''' A Record is a data structure containing the history and summary of a user's
        performance across time for a specific Question, Q.

        ----------
        Attributes
        ----------
        history_entry   =   <2-tuple>: (correct <bool>, RT <float>)
        := A pair performance outcomes for a single test iteration of Q.

        history         =   <listof history_entry>
        := A chronological list of performance outcomes.

        n_attempt       =   <int>
        := Count of times that Q has been tested.

        n_correct       =   <int>
        := Count of times that Q  has been correctly answered.

        -------
        Methods
        -------
        update_counts   :   <cls> -> <None>
        := Updates n_attempt and n_correct to reflect the contents of history.

        add_entry       :   <cls> <history_entry> -> <None>
        := Appends a history_entry to history and updates n_attempt and n_correct
           by calling update_counts.

        successes       :   <cls> -> <listof history_entry>
        := Returns a list containing correct history_entries in history.

    '''

    # Dictionary of legal assignable attributes and their mandatory types.
    legal_args = {'history': list, 'n_attempt': int, 'n_correct': int}

    def __init__(self, **kwargs):
        ''' Set attributes which are named---and, typed---correctly. '''
        legal_keys = list(self.legal_args.keys())
        for k in kwargs:
            if (k in legal_keys):
                 k_type = self.legal_args[k]
                 k_val = kwargs[k]
                 if (type(k_val) == k_type):
                    self.__setattr__(k, kwargs[k])
                 else:
                     raise TypeError("Attribute {}'s type is {} instead of {}.".format(k, type(k_val), k_type))
            else:
                raise ValueError("Attribute {} is not among legal attributes: {}".format(k, legal_keys))


    def update_counts(self, window=False):
        to_count = self.history
        total_count = len(to_count)
        if ((window) and (window >= total_count)):
            self.n_attempt = window
        else:
            self.n_attempt = total_count

        self.n_correct = len(self.successes(window))


    def add_entry(self, entry):
        if (type(entry) != tuple):
            raise TypeError('Record history entries must be tuples.')
        elif (len(entry) != 2):
            raise ValueError('Record history entries must be 2-tuples.')
        else:
            self.history.append(entry)
            self.update_counts()


    def successes(self, window=False):
        to_filter = self.history[:]
        if window:
            if (window >= len(to_filter)):
                to_filter = to_filter[-window:]
        return list(filter(lambda h: h[0], to_filter))
