from question import Question

class Database(object):
    # Dictionary of legal assignable attributes and their mandatory types.
    legal_args = {'Q': list, 'name': str}

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


    def add_q(self, q):
        if isinstance(q, Question):
            self.Q.append(q)
        else:
            raise TypeError('Only Question instances can be added to a Database.')

    def del_q(self, q):
        if isinstance(q, Question):
            if (q in self.Q):
                self.Q.pop(self.Q.index(q))
            else:
                raise ValueError('Cannot remove a non-member Question from a Database.')
        else:
            raise TypeError('Only Question instances can be removed from a Database.')

    def summarize(self):
        results = []
        for q in self.Q:
            results.append(q.summarize())


