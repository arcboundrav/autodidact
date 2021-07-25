CORRECT.wav             Audio feedback for correct responses.
INCORRECT.wav           Audio feedback for incorrect responses.

img/                    Directory containing image files used for certain questions.
pkl/                    Directory containing pickled files containing databases and questions.

record.py               Class for instantiating records tracking performance on questions.

question.py             imports record.py
                        Class for instantiating questions.

database.py             imports question.py
                        Class for instantiating databases.

util.py                 Contains helper functions for handling file names for save/load dialog box,
                        setting and preparing images to use with Cairo,
                        storing and loading pickled files,
                        and a few functions for managing probability distributions and sampling questions.

ad.py                   A version of the full software. Most up to date, includes rebalancing pmfs (50% chance).
                        Rebalancing works by redistributing mass proportional to the mass of non-modes evenly
                        across modes; this effectively makes modes even more extreme.
