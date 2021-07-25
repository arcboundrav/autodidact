from record import Record
import numpy as np

class Question(object):
    # Dictionary of legal assignable attributes and their mandatory types.
    legal_args = {'q': str, 'a': str, 'form': str, 'tags': list}

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

        # Instantiate a fresh Record for this Question
        self.refresh_record()
        # Add the 'default' tag to the tags
        self.tags.append('default')


    def refresh_record(self):
        ''' Create a fresh Record instance for this Question and reset summary stats. '''
        self.record = Record(history=[], n_attempt=0, n_correct=0)
        self.prop_correct = 0.0
        self.med_RT = 0.0


    def record_response(self, response):
        ''' Add the user's response to the Question to the record. '''
        self.record.add_entry(response)


    def summarize(self, window=False):
        ''' Compute the % of correct responses and median RT as per the record. '''
        self.record.update_counts(window=window)
        successes = self.record.successes(window=window)
        if (successes != []):
            self.prop_correct = self.record.n_correct / self.record.n_attempt
            self.med_RT = np.median(list(map(lambda s: s[1], successes)))
        else:
            self.prop_correct = 0.0
            self.med_RT = 0.0
        return (self.prop_correct, self.med_RT)

