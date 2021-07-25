import tkinter as tk
from functools import partial
from util import extant, to_pkl, from_pkl, prep_img, set_img
from record import Record
from question import Question
from database import Database
import numpy as np
import os
import time
from pygame import mixer
from scipy.stats import entropy
mixer.init(44100)

YES = mixer.Sound("CORRECT.wav")
NO = mixer.Sound("INCORRECT.wav")

BGC = 'black'
FGC = 'white'
T_FONT = ('Cambria', 24)
M_FONT = ('Cambria', 18)
S_FONT = ('Cambria', 14)


def rebalance(pmf):
    '''\
        Render modes of the distribution even more pronounced by redistributing mass proportional
        to the alternatives to modes.
    '''
    max_val = np.max(pmf)
    N = len(pmf)
    n_mode = len(pmf[pmf==max_val])
    n_donor = N - n_mode
    if (n_donor > 0):
        mode_i = []
        donation = 0.0
        for i in range(N):
            if (pmf[i]==max_val):
                mode_i.append(i)
            else:
                source = pmf[i]
                donate = (source * 3/5) * 1/n_donor
                donation += donate
                pmf[i] = source - donate
        donation = donation / n_mode
        for index in mode_i:
            pmf[index] = max_val + donation

    return pmf


def comp_dists(d0, d1, base=2):
    ''' Compares the entropy of two pmfs. '''
    H0 = entropy(d0, base=base)
    H1 = entropy(d1, base=base)
    print('H0: {}\nH1: {}'.format(H0, H1))


class App(tk.Frame):
    def __init__(self, master=None):
        self.xdim = 1280
        self.ydim = 720
        self.shim = 40
        self.DGRAY = "#686868"
        self.LGRAY = "#000000"
        self.T_FONT = ('Cambria', 24)
        self.M_FONT = ('Cambria', 18)
        self.S_FONT = ('Cambria', 14)
        self.submit_time = 0
        self.can_it_img = prep_img("LATERAL_BRODMANN_AREAS")
        self.used_reveal = False

        tk.Frame.__init__(self, master, width=self.xdim, height=self.ydim)
        self.grid_propagate(0)
        self.grid(sticky=tk.N+tk.S+tk.E+tk.W)

        self.train_props = {'window': 5, 'i': 0, 'n': 1, 'N':100, 'streak': 0, 'c': 0, 'med_RT': 'N/A', 'RT':[]}

        self.views = {'main': 0, 'q': 1, 'db': 2, 'pref': 3, 'train': 4, 'db_create': 5, 'db_edit': 6, 'init_train': 7, 'review': 8}
        self.curr_view = self.views['main']

        self.createWidgets()


    def start_training(self):
        ''' Begin to train. '''
        # Switch Views to the training screen
        self.nav_to('train')

        # Update the train_props
        #self.update_train_props()

        # Refresh the display of the train_props
        self.refresh_train_props()

        # Update the metrics to show this update
        self.update_metrics()

        # Make sure we can't submit an answer pre-maturely
        self.can_react = False

        # Sample the first question
        self.train()


    def sample_q(self):
        ''' Assign self.training_q according to the probability logic and performance history. '''
        self.gen_pdist(window=self.train_props['window'])
        self.training_q = np.random.choice(self.pdist['Q'], p=self.pdist['pmf'])
        #qi = self.pdist['QDB'].index(qi)
        #return qi


    def gen_pdist(self, window=False):
        ''' Assign self.pdist to facilitate training based on performance history. '''
        # Tell all of the Questions in the Database to update their performance history;
        # gather any questions which have never been attempted before into a separate
        # list for priority testing.
        need_ask = []
        for q in self.db_to_train.Q:
            q.summarize(window=window)
            if (q.record.n_attempt == 0):
                need_ask.append(q)

        n = float(len(need_ask))
        if n:
            # Case: There is >=1 question that has never been asked before.
            new_pmf = self.uni_dist(n)
            Q = need_ask

        else:
            # Case: All questions have been asked. Generating softmax PMF.
            need_ask = list(map(lambda q: q.prop_correct, self.db_to_train.Q))
            arr_need_ask = np.array(need_ask)

            # Inversion necessary; no normalization is standard behaviour now
            pmf = self.softmax(1 - arr_need_ask)
            new_pmf = self.softmax(1 - np.sqrt(arr_need_ask))
            if np.random.randint(2):
                print('Extremizing values')
                new_pmf = rebalance(new_pmf)
            else:
                print('No Extremizing')
            #comp_dists(pmf, new_pmf)
            #comp_dists(new_pmf, rebalanced)
            # Echo for Debugs
            for i in range(len(self.db_to_train.Q)):
                print('{}\t{}:\t{}'.format(pmf[i], new_pmf[i], self.db_to_train.Q[i].q))

            Q = self.db_to_train.Q

        self.pdist = {'Q':Q, 'pmf':new_pmf}


    def softmax(self, list_of_ratios):
        e_x = np.exp(list_of_ratios - max(list_of_ratios))
        return e_x / e_x.sum()


    def uni_dist(self, n):
        return [1./n for i in range(int(n))]


    def train(self, need_feed=False, need_wait=False):
        ''' Remember: this function controls whether answers can be submitted with self.can_react!!!
            Execute the training loop. '''
        if (self.train_props['i'] < self.train_props['N']):
            if need_feed:
                self.question.config(state=tk.NORMAL)
                self.question.insert(tk.END, self.training_q.a)
                self.question.config(state=tk.DISABLED)
                self.train(need_feed=False, need_wait=True)
            elif need_wait:
                start = time.time()
                while time.time() < start + 3:
                    print('')
                self.train()
            elif ((not need_feed) and (not need_wait)):

                # Get the question from the database to ask
                self.sample_q()
                #this_img = tk.PhotoImage(file = "sbob.gif")
                #self.this_img = prep_img("")
                # Update the question display
                self.question.config(state=tk.NORMAL)
                self.question.delete('1.0', tk.END)
                self.question.insert(tk.END, self.training_q.q)
                self.question.insert(tk.END, '\n')
                #self.question.image_create(tk.END, image = self.can_it_img)
                #self.question.image = self.can_it_img
                self.question.config(state=tk.DISABLED)

                # Unlock the response field with self.can_react
                self.can_react = True

                # Start the RT timer with self.q_RT
                self.q_RT = time.time()

        else:
            to_pkl(self.db_to_train, fn=self.db_to_train_fn, fp='./pkl/DB/')
            self.nav_to('review')


    def update_train_props(self):
        ''' Update the train_props based on the verification of an answer. '''
        self.train_props['i'] += 1
        self.train_props['n'] += 1
        self.train_props['med_RT'] = np.round(np.median(self.train_props['RT']), 2)
        self.train_props['med_RT'] = '> 9000' if (self.train_props['med_RT'] > 9000) else self.train_props['med_RT']

        if self.a_was_correct:
            self.train_props['c'] += 1
            self.train_props['streak'] += 1
        else:
            self.train_props['streak'] = 0


    def refresh_train_props(self):
        ''' Re-initialize values in the self.train_props dictionary. '''
        self.train_props['i'] = 0
        self.train_props['n'] = 1
        self.train_props['streak'] = 0
        self.train_props['c'] = 0
        self.train_props['med_RT'] = 'N/A'
        self.train_props['RT'] = []


    def update_metrics(self):
        ''' Update the Training Metrics display. '''
        self.n_out_of_N_var.set('{} / {}'.format(self.train_props['n'], self.train_props['N']))
        self.n_correct_var.set('{} / {} ({}%)'.format(self.train_props['c'], self.train_props['i'], np.round(((self.train_props['c'] / max(1, self.train_props['i'])) * 100), 2)))
        self.streak_var.set('Current Streak: {}'.format(self.train_props['streak']))
        self.med_RT_var.set('Median RT (s): {}'.format(self.train_props['med_RT']))


    def answer_react(self, a):
        ''' React to an answer. '''
        print(a)
        if self.can_react:
            # Temporarily freeze the ability to submit responses until this logic is executed
            self.can_react = False

            # Compute reaction time and store it
            self.submit_time = time.time()
            self.a_RT = self.submit_time - self.q_RT
            self.train_props['RT'].append(self.a_RT)

            # Determine if the response was correct or not
            if (isinstance(self.training_q, Question)):
                true_answer = self.training_q.a
                print(true_answer)
                if ((a == true_answer) and (not (self.used_reveal))):
                    self.a_was_correct = True
                    YES.play()
                else:
                    self.a_was_correct = False
                    NO.play()
                    self.display_answer('\n')
                    self.used_reveal = False

                    # Update the question display
                    #self.question.config(state=tk.NORMAL)
                    #self.question.delete('1.0', tk.END)
                    #self.question.insert(tk.END, '\n')
                    #self.question.insert('1.0', true_answer)
                    #self.question.config(state=tk.DISABLED)
                    #NO.play()


                # Keep track of the performance on the Question itself
                self.training_q.record_response((self.a_was_correct, self.a_RT))

                # Update the train_props as necessary
                self.update_train_props()

                # Update the metrics
                self.update_metrics()

                # Clear the entry box and return focus to it
                self.answer.delete(0, tk.END)
                self.answer.focus_set()

                # Get the next question or go to the Review View
                self.train()

            else:
                self.can_react = True


    def add_q_to_db(self):
        ''' Add a Question to a Database. '''
        if (self.focus_get() is self.q_list):
            self.q_to_add = ''
            self.q_to_add_fn = ''
            self.q_to_add = self.q_list.get(tk.ACTIVE)
            if ((self.q_to_add != '') and (isinstance(self.db_to_edit, Database))):
                filename_end_i = self.q_to_add.index(':')
                self.q_to_add_fn = self.q_to_add[:filename_end_i]
                self.q_to_add = from_pkl(self.q_to_add_fn, fp='./pkl/Q/')
                self.db_to_edit.add_q(self.q_to_add)
                to_pkl(self.db_to_edit, self.db_to_edit_fn, fp='./pkl/DB/')
                self.db_to_edit = from_pkl(self.db_to_edit_fn, fp='./pkl/DB/')
                self.display_dbq()


    def rem_q_from_db(self):
        ''' Remove a Question from a Database. '''
        if (self.focus_get() is self.dbq_list):
            self.q_to_rem = ''
            self.q_to_rem = self.dbq_list.get(tk.ACTIVE)

            if (self.q_to_rem != ''):
                # Remove the '> "' and '..."' at the beginning and end of the string
                # when applicable.
                if (self.q_to_rem[:3] == '> "'):
                    self.q_to_rem = self.q_to_rem[3:]

                if (self.q_to_rem[-1] == '"'):
                    self.q_to_rem = self.q_to_rem[:-1]

                if (len(self.q_to_rem) > 3):
                    if (self.q_to_rem[-3:] == '...'):
                        self.q_to_rem = self.q_to_rem[:-3]

                # Now that we have the string...
                # Load up all of the Questions in the database, pair them with their question text,
                # and filter the list to only include Questions with question text that doesn't
                # contain the substring given by self.q_to_rem. This is only going to be buggy if
                # they try to remove a question which is identical up to the first 50 characters.
                question_copy = self.db_to_edit.Q[:]
                q_tups = [(question, question.q) for question in question_copy]
                filtered_question_copy = list(filter(lambda qtup: (not (self.q_to_rem in qtup[1])), q_tups))
                filtered_Q = list(map(lambda qtup: (qtup[0]), filtered_question_copy))
                self.db_to_edit.Q = filtered_Q

                # Now that the db_to_edit has had the question removed, update it on disk and then display
                # the updated object.
                to_pkl(self.db_to_edit, self.db_to_edit_fn, fp='./pkl/DB/')
                self.db_to_edit = from_pkl(self.db_to_edit_fn, fp='./pkl/DB/')
                self.display_dbq()
                self.dbq_list.selection_set(0)


    def is_empty(self):
        ''' Verify whether or not the loaded training database is empty or not. '''
        if (len(self.db_to_train.Q) > 0):
            return False
        else:
            return True


    def init_react(self, selected=None):
        ''' Figure out which database we are using for Training and load it. '''
        self.db_to_train = selected
        if self.db_to_train:
            last_i = selected.index(':')
            self.db_to_train_fn = selected[:(last_i)]
            self.db_to_train = from_pkl(self.db_to_train_fn, fp='./pkl/DB/')
            self.training_q = None


    def db_react(self, selected=None):
        ''' Figure out which database we picked in the OptionMenu for editting. '''
        self.db_to_edit = selected
        if self.db_to_edit:
            last_i = selected.index(':')
            self.db_to_edit_fn = selected[:(last_i)]
            self.db_to_edit = from_pkl(self.db_to_edit_fn, fp='./pkl/DB/')
            self.display_dbq()


    def validate_question(self):
        ''' React to an attempt to create a Question. '''
        id_it = False
        q_text = self.q_text.get()
        a_text = self.a_text.get()
        if ((q_text != "") and (a_text != '')):
            try:
                new_q = Question(q=q_text, a=a_text, form='direct', tags=['default'])
                id_it = True
            except:
                print('Illegal inputs for Question detected.')

            finally:
                if id_it:
                    self.auto_id_q(new_q)
                    self.q_text.delete(0, tk.END)
                    self.a_text.delete(0, tk.END)
                    self.display_questions()
                    self.q_text.focus_set()
                else:
                    print('Failed to save the question!')


    def validate_database(self):
        ''' React to an attempt to create a Database. '''
        id_it = False
        name_text = self.db_text.get()
        if (name_text != ""):
            try:
                new_db = Database(Q=[], name=name_text)
                id_it = True
            except:
                print('Illegal inputs for Database detected.')

            finally:
                if id_it:
                    self.auto_id_db(new_db)
                    self.db_text.delete(0, tk.END)
                    self.display_databases()
                    self.db_text.focus_set()

                else:
                    print('Failed to save the database!')


    def list_questions(self):
        ''' Return a list of the question pickles stored in './pkl/Q/' '''
        curr_q_list = sorted(list(enumerate(os.listdir('./pkl/Q/'))), key=lambda x: x[0])
        curr_q_list = list(map(lambda t: t[1], curr_q_list))
        return curr_q_list


    def list_dbq(self):
        ''' Return a list of the questions stored in a Database '''
        if (isinstance(self.db_to_edit, Database)):
            dbq_list = self.db_to_edit.Q
            dbq_list = [dbq.q for dbq in dbq_list]
            dbq_list = [dbq[:50]+'...' if len(dbq)>50 else dbq for dbq in dbq_list]
        else:
            dbq_list = []
        return dbq_list


    def list_dbq_train(self):
        pass


    def display_dbq(self):
        ''' Update the displayed Listbox of the to-edit Database Questions. '''
        self.dbq_list.delete(0, tk.END)
        curr_dbq_list = self.list_dbq()
        if (curr_dbq_list != []):
            for dbq in curr_dbq_list:
                this_dbq_text = '"{}"'.format(dbq)
                self.dbq_list.insert(tk.END, '> {}'.format(this_dbq_text))
            #self.dbq_list.selection_set(0)


    def list_databases(self):
        """ Return a list of the database pickles stored in './pkl/DB/' """
        curr_db_list = os.listdir('./pkl/DB/')
        return curr_db_list


    def auto_id_db(self, db):
        ''' Pickle a created Database and automatically index it. '''
        curr_db_list = self.list_databases()
        next_db_i = len(curr_db_list)
        need_id = True
        while need_id:
            fn = 'db{}'.format(next_db_i)
            if extant(fn, fp='./pkl/DB/'):
                next_db_i += 1
            else:
                to_pkl(db, 'db{}'.format(next_db_i), fp='./pkl/DB/')
                need_id = False


    def auto_id_q(self, q):
        ''' Pickle a created Question and automatically index it. '''
        curr_q_list = self.list_questions()
        next_q_i = len(curr_q_list)
        need_id = True
        while need_id:
            fn = 'q{}'.format(next_q_i)
            if extant(fn, fp='./pkl/Q/'):
                next_q_i += 1
            else:
                to_pkl(q, 'q{}'.format(next_q_i), fp='./pkl/Q/')
                need_id = False


    def display_questions(self):
        ''' Update the displayed Listbox of pickled Questions. '''
        self.q_list.delete(0, tk.END)
        curr_q_list = self.list_questions()
        if (curr_q_list != []):
            sorted_list = []
            for q in curr_q_list:
                this_q = from_pkl(q[:-2], fp='./pkl/Q/')
                this_q_text = this_q.q
                this_q_text = this_q_text[:50]+'...' if len(this_q_text)>50 else this_q_text
                this_q_text = '{}:\t\t{}"'.format(q[:-2], this_q_text)
                sorted_list.append(this_q_text)
            sorted_list = sorted(sorted_list)
            for this_q_text in sorted_list:
                self.q_list.insert(tk.END, '{}'.format(this_q_text))
            #self.q_list.selection_set(0)


    def display_databases(self):
        ''' Update the displayed Listbox of pickled Databases. '''
        print("DEBUG | display_databases() is being called.")
        self.db_list.delete(0, tk.END)
        curr_db_list = self.list_databases()
        if (curr_db_list != []):
            sorted_list = []
            for db in curr_db_list:
                this_db = from_pkl(db[:-2], fp='./pkl/DB/')
                this_db_text = this_db.name
                this_db_text = this_db_text[:50]+'...' if len(this_db_text)>50 else this_db_text
                # NOTE # The following line was missing, and so selecting the correct db file
                # by name wasn't working.
                this_db_text = '{}:\t\t{}'.format(db[:-2], this_db_text)
                sorted_list.append(this_db_text)
            sorted_list = sorted(sorted_list)
            for this_db_text in sorted_list:
                self.db_list.insert(tk.END, '{}'.format(this_db_text))
            #self.db_list.selection_set(0)


    def attempt_deletion(self):
        ''' Command attached to delete button. '''
        # Get the selected item from the Listbox
        target_q = self.q_list.get(tk.ACTIVE)
        # Such strings are in the form:
        # 'q#:\t\t"blah..."'
        # Therefore, the filename is the substring of target_q up until the first
        # colon; self.delete_question already has default parameters for hitting
        # the correct filepath and file extension, so, we just need this string.
        if (target_q != ''):
            filename_end_i = target_q.index(':')
            target_q_filename = target_q[:filename_end_i]
            self.delete_question(target_q_filename)


    def delete_question(self, fn, fp='./pkl/Q/', fx='.p'):
        filename = fp + fn + fx
        if extant(fn, fp=fp):
            os.remove(filename)
            self.display_questions()
            self.q_list.selection_set(0)


    def attempt_db_deletion(self):
        ''' Command attached to delete button. '''
        # Get the selected item from the Listbox
        target_db = self.db_list.get(tk.ACTIVE)
        # Such strings are in the form: 'db#:\t\t"blah..."'
        # Therefore, the filename is the substring of target_db up until the first colon;
        # self.delete_database already has default parameters for hitting the correct
        # filepath and file extension, so, we just need this string.
        if (target_db != ''):
            filename_end_i = target_db.index(':')
            target_db_filename = target_db[:filename_end_i]
            self.delete_database(target_db_filename)


    def delete_database(self, fn, fp='./pkl/DB/', fx='.p'):
        filename = fp + fn + fx
        if extant(fn, fp=fp):
            os.remove(filename)
            self.display_databases()
            self.db_list.selection_set(0)
        else:
            print("filename <{}>: {} | Database allegedly doesn't exist".format(type(fn), fn))


    def display_answer(self, n):
        self.used_reveal = True
        self.question.config(state=tk.NORMAL)
        self.question.tag_config("HI", background="#000000", foreground='#00ff00')
        #img_to_disp = prep_img('BG')
        #self.question.image_create(tk.END, image = img_to_disp)
        self.question.insert(tk.END, n)
        self.question.insert(tk.END, 'Correct Answer:')
        self.question.insert(tk.END, n)
        self.question.insert(tk.END, '\t')
        self.question.insert(tk.END, self.training_q.a, ("HI"))
        self.question.config(state=tk.DISABLED)


    def nav_to(self, view):
        self.curr_view = self.views[view]
        self.main.destroy()
        self.createWidgets()


    def createWidgets(self):

        self.main = tk.Frame(self, width=self.xdim, height=self.ydim)
        if (self.curr_view == self.views['q']):

            # Title of the Main Frame
            self.title_text = tk.StringVar()
            self.title_text.set('Create a Question')
            self.title = tk.Label(self.main, textvariable=self.title_text, fg=BGC, font=T_FONT)
            self.title.grid(row=0, column=1, sticky=tk.N)

            # Text Field Labels
            self.q_text_label = tk.Label(self.main, text='Question Text:', font=M_FONT)
            self.q_text_label.grid(row=1, column=0, sticky=tk.E)

            self.a_text_label = tk.Label(self.main, text='Answer Text:', font=M_FONT)
            self.a_text_label.grid(row=2, column=0, sticky=tk.E)


            # Text Fields
            self.q_text = tk.Entry(self.main, width=100)
            self.q_text.grid(row=1, column=1, columnspan=2,  sticky=tk.W)

            self.a_text = tk.Entry(self.main, width=100)
            self.a_text.grid(row=2, column=1, columnspan=2, sticky=tk.W)


            # Submission Button
            self.submit_q = tk.Button(self.main, text='Submit', font=M_FONT, command=self.validate_question)
            self.submit_q.grid(row=5, column=1, sticky=tk.S)

            # Listbox to display the questions in the ./pkl/Q/ subdirectory
            ### Prepare addition of a vertical scrollbar
            self.q_list_y_scroll = tk.Scrollbar(self.main, orient=tk.VERTICAL)
            self.q_list_y_scroll.grid(row=6, column=2, sticky=tk.N+tk.S+tk.W)

            self.q_list = tk.Listbox(self.main, font=S_FONT, yscrollcommand=self.q_list_y_scroll.set, selectmode=tk.SINGLE, selectforeground='yellow', activestyle='dotbox')
            self.q_list.grid(row=6, column=0, columnspan=2, sticky=tk.W+tk.S+tk.N+tk.E)
            self.q_list_y_scroll['command'] = self.q_list.yview
            self.display_questions()

            # Delete Button
            self.delete_q = tk.Button(self.main, text='Delete', font=M_FONT, command=self.attempt_deletion)
            self.delete_q.grid(row=6, column=2)

            # Navigation Button
            self.nav_to_main = tk.Button(self.main, text='Return to Main Menu', font=M_FONT, command=partial(self.nav_to, 'main'))
            self.nav_to_main.grid(row=0, column=2)

            # Place main inside the app and give the correct widget focus
            self.main.grid(sticky=tk.N+tk.S+tk.E+tk.W)
            self.q_text.focus_set()


        elif (self.curr_view == self.views['main']):

            # Title of the Main Frame
            self.title_text = tk.StringVar()
            self.title_text.set('Main Menu')
            self.title = tk.Label(self.main, textvariable=self.title_text, fg=BGC, font=T_FONT)
            self.title.grid(row=0)

            # Frame to Hold Navigation Buttons
            self.nav_frame = tk.Frame(self.main)

            # Navigation Button -> Questions (curr_view -> 1)
            self.nav_to_q_button = tk.Button(self.nav_frame, text='Manage Questions', font=M_FONT, command=partial(self.nav_to, 'q'))
            self.nav_to_q_button.grid(row=0, sticky=tk.E+tk.W)

            # Navigation Button -> Databases (curr_view -> 2)
            self.nav_to_db_button = tk.Button(self.nav_frame, text='Manage Databases', font=M_FONT, command=partial(self.nav_to, 'db'))
            self.nav_to_db_button.grid(row=1, sticky=tk.E+tk.W)

            # Navigation Button -> Preferences (curr_view -> 3)
            self.nav_to_pref_button = tk.Button(self.nav_frame, text='Preferences', font=M_FONT, command=partial(self.nav_to, 'pref'))
            self.nav_to_pref_button.grid(row=2, sticky=tk.E+tk.W)

            # Navigation Button -> Training (curr_view -> 4)
            self.nav_to_train_button  = tk.Button(self.nav_frame, text='Training', font=M_FONT, command=partial(self.nav_to, 'init_train'))
            self.nav_to_train_button.grid(row=3, sticky=tk.E+tk.W)

            # Exit Button
            self.exit = tk.Button(self.nav_frame, text='Exit', font=M_FONT, command=self.quit)
            self.exit.grid(row=4, sticky=tk.W+tk.E)

            # Place nav_frame inside main
            self.nav_frame.grid(row=1, sticky=tk.N+tk.S+tk.E+tk.W)
            #self.nav_frame.place(in_=self.main, anchor='c', relx=0.5, rely=0.5)

            # Place main inside the app
            self.main.grid(sticky=tk.N+tk.S+tk.E+tk.W)

        elif (self.curr_view == self.views['db']):

            # Title
            self.title = tk.Label(self.main, text='Manage Databases', fg=BGC, font=T_FONT)
            self.title.grid(row=0)

            # Frame to Hold Navigation Buttons
            self.nav_frame = tk.Frame(self.main)

            # Navigation Button -> Create/Destroy Databases (curr_view -> 5)
            self.nav_to_q_button = tk.Button(self.nav_frame, text='Add / Remove Databases', font=M_FONT, command=partial(self.nav_to, 'db_create'))
            self.nav_to_q_button.grid(row=0, sticky=tk.E+tk.W)

            # Navigation Button -> Edit Databases (curr_view -> 6)
            self.nav_to_db_button = tk.Button(self.nav_frame, text='Edit Databases', font=M_FONT, command=partial(self.nav_to, 'db_edit'))
            self.nav_to_db_button.grid(row=1, sticky=tk.E+tk.W)

            # Navigation Button -> Main Menu (curr_view -> 0)
            self.nav_to_main = tk.Button(self.nav_frame, text='Return to Main Menu', fg=BGC, font=M_FONT, command=partial(self.nav_to, 'main'))
            self.nav_to_main.grid(row=2, sticky=tk.E+tk.W)

            # Place nav_frame inside main
            self.nav_frame.grid(row=1, sticky=tk.N+tk.S+tk.E+tk.W)

            # Place main inside the app
            self.main.grid(sticky=tk.N+tk.S+tk.E+tk.W)


        elif (self.curr_view == self.views['pref']):

            # Title
            self.title = tk.Label(self.main, text='Edit Preferences', fg=BGC, font=T_FONT)
            self.title.grid(row=0)

            # Navigation Button -> Main Menu (curr_view -> 0)
            self.nav_to_main = tk.Button(self.main, text='Return to Main Menu', fg=BGC, font=T_FONT, command=partial(self.nav_to, 'main'))
            self.nav_to_main.grid(row=1)

            # Place main inside the app
            self.main.grid(sticky=tk.N+tk.S+tk.E+tk.W)

        elif (self.curr_view == self.views['review']):

            # Title
            self.title = tk.Label(self.main, text='Review Screen', fg=BGC, font=T_FONT)
            self.title.grid(row=0)

            # Navigation Button -> Main Menu (curr_view -> 0)
            self.nav_to_main = tk.Button(self.main, text='Return to Main Menu', fg=BGC, font=T_FONT, command=partial(self.nav_to, 'main'))
            self.nav_to_main.grid(row=1)

            # Place main inside the app
            self.main.grid(sticky=tk.N+tk.S+tk.E+tk.W)


        elif (self.curr_view == self.views['init_train']):

            # Title
            self.title = tk.Label(self.main, text='Initialize Training Session', fg=BGC, font=T_FONT)
            self.title.grid(row=0)

            # Navigation Button -> Main Menu (curr_view -> 0)
            self.nav_to_main = tk.Button(self.main, text='Return to Main Menu', fg=BGC, font=T_FONT, command=partial(self.nav_to, 'main'))
            self.nav_to_main.grid(row=1)

            # OptionMenu Label
            self.db_menu_label = tk.Label(self.main, text='Select Database Source for Training:', fg=BGC, font=S_FONT)
            self.db_menu_label.grid(row=2, column=0, sticky=tk.E)

            # OptionBox for Displaying Databases
            ### Create a control variable first
            self.db_menu_var = tk.StringVar()
            self.list_of_db = []
            self.list_of_db.extend(self.list_databases())

            # Make sure we don't list empty databases
            self.non_empty_db = []
            if (self.list_of_db != []):
                for db in self.list_of_db:
                    this_db = from_pkl(db[:-2], fp='./pkl/DB/')
                    if (len(this_db.Q) > 0):
                        self.non_empty_db.append(db)
                self.list_of_db = self.non_empty_db

            self.freeze_db_menu = False

#            if (self.list_questions() == []):
#                self.db_menu_var.set(('No questions on file. Create one first.'))
#                self.list_of_db = ('blah', 'blah')
#                self.freeze_db_menu = True

            if (self.list_of_db != []):
                db_names = []
                for db in self.list_of_db:
                    this_db = from_pkl(db[:-2], fp='./pkl/DB/')
                    this_db_text = this_db.name
                    this_db_text = this_db_text[:50]+'...' if len(this_db_text)>50 else this_db_text
                    db_names.append('{}: {}'.format(db[:-2], this_db_text))
                self.list_of_db = tuple(db_names)
                self.db_menu_var.set(self.list_of_db[0])

            else:
                self.db_menu_var.set(('No databases with questions on file.'))
                self.list_of_db = ('placeholder', 'placeholder')
                self.freeze_db_menu = True

            self.db_menu = tk.OptionMenu(self.main, self.db_menu_var, *self.list_of_db, command=self.init_react)
            self.db_menu.config(width=30)

            if self.freeze_db_menu:
                self.db_menu.config(state=tk.DISABLED)
            else:
                # Assume the first thing in the list is automatically selected for training
                self.init_react(self.db_menu_var.get())

            self.db_menu.grid(row=2, column=1, sticky=tk.W)

            self.begin = tk.Button(self.main, text='Begin Training', command=self.start_training)
            self.begin.grid(row=3, sticky=tk.W)

            if self.freeze_db_menu:
                self.begin.config(state=tk.DISABLED)

            # Place main inside the app
            self.main.grid(sticky=tk.N+tk.S+tk.E+tk.W)


        elif (self.curr_view == self.views['train']):

            self.top = tk.Frame(self.main, width=self.xdim, height=self.shim*1, bg=self.DGRAY)
            self.tmid = tk.Frame(self.main, width=self.xdim, height=self.shim*9, bg="#808080")
            self.mid = tk.Frame(self.main, width=self.xdim, height=self.shim*1, bg=self.DGRAY)
            self.bmid = tk.Frame(self.main, width=self.xdim, height=self.shim*6, bg="#808080")
            self.bot = tk.Frame(self.main, width=self.xdim, height=self.shim*1, bg=self.DGRAY)

            self.top.grid_propagate(0)
            self.tmid.grid_propagate(0)
            self.mid.grid_propagate(0)
            self.bmid.grid_propagate(0)
            self.bot.grid_propagate(0)

            self.top.grid(row=0, column=0, columnspan=32, sticky=tk.W+tk.E+tk.N+tk.S)
            self.tmid.grid(row=1, column=0, columnspan=32, sticky=tk.W+tk.E+tk.N+tk.S)
            self.mid.grid(row=2, column=0, columnspan=32, sticky=tk.W+tk.E+tk.N+tk.S)
            self.bmid.grid(row=3, column=0, columnspan=32, sticky=tk.W+tk.E+tk.N+tk.S)
            self.bot.grid(row=4, column=0, columnspan=32, sticky=tk.W+tk.E+tk.N+tk.S)

            self.tm_1 = tk.Frame(self.tmid, width=self.shim*1, height=self.shim*9, bg=self.DGRAY)
            self.tm_2 = tk.Frame(self.tmid, width=self.shim*23, height=self.shim*9, bg="#808080")
            self.tm_3 = tk.Frame(self.tmid, width=self.shim*1, height=self.shim*9, bg=self.DGRAY)
            self.tm_4 = tk.Frame(self.tmid, width=self.shim*6, height=self.shim*9, bg="#808080")
            self.tm_5 = tk.Frame(self.tmid, width=self.shim*1, height=self.shim*9, bg=self.DGRAY)


            self.tm_1.grid_propagate(0)
            self.tm_2.grid_propagate(0)
            self.tm_3.grid_propagate(0)
            self.tm_4.grid_propagate(0)
            self.tm_5.grid_propagate(0)

            self.bm_1 = tk.Frame(self.bmid, width=self.shim*1, height=self.shim*6, bg=self.DGRAY)
            self.bm_2 = tk.Frame(self.bmid, width=self.shim*23, height=self.shim*6, bg="#808080")
            self.bm_3 = tk.Frame(self.bmid, width=self.shim*1, height=self.shim*6, bg=self.DGRAY)
            self.bm_4 = tk.Frame(self.bmid, width=self.shim*6, height=self.shim*6, bg="#808080")
            self.bm_5 = tk.Frame(self.bmid, width=self.shim*1, height=self.shim*6, bg=self.DGRAY)

            self.bm_1.grid_propagate(0)
            self.bm_2.grid_propagate(0)
            self.bm_3.grid_propagate(0)
            self.bm_4.grid_propagate(0)
            self.bm_5.grid_propagate(0)

            self.tm_1.grid(row=0, column=0, columnspan=1, sticky=tk.W+tk.E+tk.N+tk.S)
            self.tm_2.grid(row=0, column=1, columnspan=23, sticky=tk.W+tk.E+tk.N+tk.S)
            self.tm_3.grid(row=0, column=24, columnspan=1, sticky=tk.W+tk.E+tk.N+tk.S)
            self.tm_4.grid(row=0, column=25, columnspan=6, sticky=tk.W+tk.E+tk.N+tk.S)
            self.tm_5.grid(row=0, column=31, columnspan=1, sticky=tk.W+tk.E+tk.N+tk.S)

            self.bm_1.grid(row=0, column=0, columnspan=1, sticky=tk.W+tk.E+tk.N+tk.S)
            self.bm_2.grid(row=0, column=1, columnspan=23, sticky=tk.W+tk.E+tk.N+tk.S)
            self.bm_3.grid(row=0, column=24, columnspan=1, sticky=tk.W+tk.E+tk.N+tk.S)
            self.bm_4.grid(row=0, column=25, columnspan=6, sticky=tk.W+tk.E+tk.S+tk.N)
            self.bm_5.grid(row=0, column=31, columnspan=1, sticky=tk.W+tk.E+tk.N+tk.S)

            # Contents of bm_4 (Restart and Exit Buttons)
            self.test = tk.Frame(self.bm_4, bg=self.master.cget('bg'))
            self.test.place(in_=self.bm_4, anchor='c', relwidth=0.95, relheight=0.95, relx=0.5, rely=0.5)

            self.restart = tk.Button(self.test, text='Reveal', font=self.M_FONT, bg=self.master.cget('bg'), command=partial(self.display_answer, '\n'))
            self.restart.place(in_=self.test, anchor='c', relwidth=1.0, relheight=0.5, relx=0.5, rely=0.25)

            self.exit = tk.Button(self.test, text='Exit', font=self.M_FONT, bg=self.master.cget('bg'), command=partial(self.nav_to, 'main'))
            self.exit.place(in_=self.test, anchor='c', relwidth=1.0, relheight=0.5, relx=0.5, rely=0.75)

            # Contents of tm_4 (Labels Displaying Progress and Cumulative Performance Metrics)
            self.metrics = tk.Frame(self.tm_4, bg=self.master.cget('bg'))
            self.metrics.place(in_=self.tm_4, anchor='c', relwidth=0.95, relheight=0.975, relx=0.5, rely=0.5)

            self.n_out_of_N_var = tk.StringVar()
            self.n_out_of_N_var.set('{} / {}'.format(self.train_props['n'], self.train_props['N']))
            self.n_out_of_N = tk.Label(self.metrics, textvariable=self.n_out_of_N_var, font=self.S_FONT, bg=self.master.cget('bg'))
            self.n_out_of_N.place(in_=self.metrics, anchor='n', relwidth=1.0, relheight=0.25, relx=0.5, rely=0)

            self.n_correct_var = tk.StringVar()
            self.n_correct_var.set('{} / {} ({}%)'.format(self.train_props['c'], (self.train_props['i'] + 1), self.train_props['c'] / (self.train_props['i'] + 1)))
            self.n_correct_label = tk.Label(self.metrics, textvariable=self.n_correct_var, font=self.S_FONT, bg=self.master.cget('bg'))
            self.n_correct_label.place(in_=self.metrics, anchor='n', relwidth=1.0, relheight=0.25, relx=0.5, rely=0.25)

            self.streak_var = tk.StringVar()
            self.streak_var.set('Current Streak: {}'.format(self.train_props['streak']))
            self.streak_label = tk.Label(self.metrics, textvariable=self.streak_var, font=self.S_FONT, bg=self.master.cget('bg'))
            self.streak_label.place(in_=self.metrics, anchor='n', relwidth=1.0, relheight=0.25, relx=0.5, rely=0.5)

            self.med_RT_var = tk.StringVar()
            self.med_RT_var.set('Median RT (ms): {}'.format(self.train_props['med_RT']))
            self.med_RT_label = tk.Label(self.metrics, textvariable=self.med_RT_var, font=self.S_FONT, bg=self.master.cget('bg'))
            self.med_RT_label.place(in_=self.metrics, anchor='n', relwidth=1.0, relheight=0.25, relx=0.5, rely=0.75)


            # Contents of bm_2 (Input for Responses)
            self.responses = tk.Frame(self.bm_2)
            self.responses.place(in_=self.bm_2, anchor='c', relwidth=0.98, relheight=0.95, relx=0.5, rely=0.5)

            self.response_label = tk.Label(self.responses, text='Answer:  ', font=self.M_FONT)
            self.response_label.place(in_=self.responses, anchor='w', relwidth=0.20, relheight=1, rely=0.5)

            self.answer = tk.Entry(self.responses, font=self.S_FONT)
            self.answer.place(in_=self.responses, anchor='w', relwidth=0.70, relheight=0.25, relx=0.2, rely=0.5)
            self.answer.focus_set()
            self.answer.bind("<Return>", (lambda event: self.answer_react(self.answer.get())))

            self.answer_submit = tk.Button(self.responses, text='Submit', font=self.M_FONT, command=(lambda: self.answer_react(self.answer.get())))
            self.answer_submit.place(in_=self.responses, anchor='w', relwidth=0.1, relheight=0.25, relx=0.5, rely=0.80)

            # Contents of tm_2 (Question Display)
            self.questions = tk.Frame(self.tm_2, bg=self.master.cget('bg'))
            self.questions.place(in_=self.tm_2, anchor='c', relwidth=0.98, relheight=0.95, relx=0.5, rely=0.5)

            self.question_label = tk.Label(self.questions, text='Question:', font=self.M_FONT)
            self.question_label.place(in_=self.questions, anchor='w', relwidth=0.20, relheight=1, rely=0.15)

            self.question = tk.Text(self.questions, fg='black', bg=self.master.cget('bg'), bd=0, font=self.M_FONT, exportselection=0, wrap=tk.WORD, height=8, highlightthickness=0, highlightbackground=self.master.cget('bg'), highlightcolor='black', selectborderwidth=0, selectbackground=self.master.cget('bg'), selectforeground='black', takefocus=0)
            self.question.place(in_=self.questions, anchor='w', relx=0.20, rely=0.55, relwidth=0.70)
            self.question.insert(tk.END, 'This question should not be available for view.')
            self.question.config(state=tk.DISABLED)

            # Title
#            self.title = tk.Label(self.main, text='Training', fg=BGC, font=T_FONT)
#            self.title.grid(row=0)

            # Question Text Box
#            self.q_text = tk.StringVar()
#            self.q_text.set('Question text here!')
#            self.q_box = tk.Label(self.main, textvariable=self.q_text, fg=BGC, font=M_FONT)
#            self.q_box.grid(row=2, columnspan=2, sticky=tk.W+tk.E+tk.N+tk.S)

            # Question Response Box
#            self.a_text = tk.StringVar()
#            self.a_text.set('')
#            self.a_box = tk.Entry(self.main, width=100)
#            self.a_box.grid(row=3, columnspan=2, sticky=tk.W+tk.E+tk.N+tk.S)

            # Navigation Button -> Main Menu (curr_view -> 0)
#            self.nav_to_main = tk.Button(self.main, text='Return to Main Menu', fg=BGC, font=T_FONT, command=partial(self.nav_to, 'main'))
#            self.nav_to_main.grid(row=1)

            # Place main inside the app
            self.main.grid(sticky=tk.N+tk.S+tk.E+tk.W)
#            self.a_box.focus_set()

        elif (self.curr_view == self.views['db_create']):

            # Title
            self.title = tk.Label(self.main, text='Add / Remove Databases', fg=BGC, font=T_FONT)
            self.title.grid(row=0, columnspan=3)

            # Text Field Labels
            self.db_text_label = tk.Label(self.main, text='Database Name:', font=M_FONT)
            self.db_text_label.grid(row=2, column=0, sticky=tk.E)

            # Text Fields
            self.db_text = tk.Entry(self.main, width=100)
            self.db_text.grid(row=2, column=1, columnspan=2,  sticky=tk.W)

            # Submission Button
            self.submit_q = tk.Button(self.main, text='Submit', font=M_FONT, command=self.validate_database)
            self.submit_q.grid(row=3, column=1, sticky=tk.S)

            # Listbox to display the questions in the ./pkl/Q/ subdirectory
            ### Prepare addition of a vertical scrollbar
            self.db_list_y_scroll = tk.Scrollbar(self.main, orient=tk.VERTICAL)
            self.db_list_y_scroll.grid(row=4, column=2, sticky=tk.N+tk.S+tk.W)

            self.db_list = tk.Listbox(self.main, font=S_FONT, yscrollcommand=self.db_list_y_scroll.set, selectmode=tk.SINGLE, selectforeground='yellow', activestyle='dotbox')
            self.db_list.grid(row=4, column=0, columnspan=2, sticky=tk.W+tk.S+tk.N+tk.E)
            self.db_list_y_scroll['command'] = self.db_list.yview
            self.display_databases()

            # Delete Button
            self.delete_db = tk.Button(self.main, text='Delete', font=M_FONT, command=self.attempt_db_deletion)
            self.delete_db.grid(row=4, column=2)

            # Navigation Button -> Manage Databases
            self.nav_to_db = tk.Button(self.main, text='Return to Manage Databases', fg=BGC, font=T_FONT, command=partial(self.nav_to, 'db'))
            self.nav_to_db.grid(row=1)

            # Place main inside the app
            self.main.grid(sticky=tk.N+tk.S+tk.E+tk.W)


        elif (self.curr_view == self.views['db_edit']):

            # Title
            self.title = tk.Label(self.main, text='Edit Databases', fg=BGC, font=T_FONT)
            self.title.grid(row=0)

            # Navigation Button -> Manage Databases
            self.nav_to_db = tk.Button(self.main, text='Return to Manage Databases', fg=BGC, font=M_FONT, command=partial(self.nav_to, 'db'))
            self.nav_to_db.grid(row=1)

            # OptionMenu Label
            self.db_menu_label = tk.Label(self.main, text='Select a Database to Edit:', fg=BGC, font=S_FONT)
            self.db_menu_label.grid(row=2, column=0, sticky=tk.E)

            # OptionBox for Displaying Databases
            ### Create a control variable first
            self.db_menu_var = tk.StringVar()
            self.list_of_db = []
            self.list_of_db.extend(self.list_databases())

            self.freeze_db_menu = False

            if (self.list_questions() == []):
                self.db_menu_var.set(('No questions on file. Create one first.'))
                self.list_of_db = ('blah', 'blah')
                self.freeze_db_menu = True

            elif (self.list_of_db != []):
                db_names = []
                for db in self.list_of_db:
                    this_db = from_pkl(db[:-2], fp='./pkl/DB/')
                    this_db_text = this_db.name
                    this_db_text = this_db_text[:50]+'...' if len(this_db_text)>50 else this_db_text
                    db_names.append('{}: {}'.format(db[:-2], this_db_text))
                self.list_of_db = tuple(db_names)
                self.db_menu_var.set(self.list_of_db[0])

            else:
                self.db_menu_var.set(('No databases on file. Create one first.'))
                self.list_of_db = ('placeholder', 'placeholder')
                self.freeze_db_menu = True

            self.db_menu = tk.OptionMenu(self.main, self.db_menu_var, *self.list_of_db, command=self.db_react)
            self.db_menu.config(width=30)

            if self.freeze_db_menu:
                self.db_menu.config(state=tk.DISABLED)

            self.db_menu.grid(row=2, column=1, sticky=tk.W)

            # Frame to Handle the Listboxes and buttons for the addition/removal of Questions to a selected
            # Database
#            self.bottom_frame = tk.Frame(self.main)
#            self.bottom_frame.grid(row=3, columnspan=4)

            # Listboxes and stuff
            # Listbox to display a Database's questions
            ### Prepare addition of a vertical scrollbar
            self.dbq_list_y_scroll = tk.Scrollbar(self.main, orient=tk.VERTICAL)
            self.dbq_list_y_scroll.grid(row=3, column=2, sticky=tk.N+tk.S+tk.W)

            self.dbq_list = tk.Listbox(self.main, font=S_FONT, yscrollcommand=self.dbq_list_y_scroll.set, selectmode=tk.SINGLE, selectforeground='yellow', activestyle='dotbox')
            self.dbq_list.grid(row=3, column=0, columnspan=2, sticky=tk.W+tk.S+tk.N+tk.E)
            self.dbq_list_y_scroll['command'] = self.dbq_list.yview

            if self.freeze_db_menu:
                self.dbq_list.config(state=tk.DISABLED)
            else:
                # Automatically assume first option in optionmenu is up for editting
                self.db_react(self.db_menu_var.get())

            # Handle the buttons between the Listboxes
            self.button_frame = tk.Frame(self.main)
            self.button_frame.grid(row=3, column=3)

            self.add = tk.Button(self.button_frame, text='Add', command=self.add_q_to_db)
            self.add.grid(row=0)

            self.remove = tk.Button(self.button_frame, text='Remove', command=self.rem_q_from_db)
            self.remove.grid(row=1)


            # Listbox to display the questions in the ./pkl/Q/ subdirectory
            ### Prepare addition of a vertical scrollbar
            self.q_list_y_scroll = tk.Scrollbar(self.main, orient=tk.VERTICAL)
            self.q_list_y_scroll.grid(row=3, column=6, sticky=tk.N+tk.S+tk.W)

            self.q_list = tk.Listbox(self.main, font=S_FONT, yscrollcommand=self.q_list_y_scroll.set, selectmode=tk.SINGLE, selectforeground='yellow', activestyle='dotbox')
            self.q_list.grid(row=3, column=4, columnspan=2, sticky=tk.W+tk.S+tk.N+tk.E)
            self.q_list_y_scroll['command'] = self.q_list.yview

            if self.freeze_db_menu:
                self.q_list.config(state=tk.DISABLED)
            else:
                self.display_questions()

            # Place main inside the app
            self.main.grid(sticky=tk.N+tk.S+tk.E+tk.W)



root = tk.Tk()
#root.wm_attributes('-type', 'splash')
#root.attributes('-zoomed', True)
app = App(master=root)
app.master.title('Autodidact')

#########################
# Debug command functions
#########################
def end_it(event):
    root.destroy()

def switch_view(event):
    if (app.curr_view == 0):
        app.curr_view = 1
        app.main.destroy()
    elif (app.curr_view == 1):
        app.curr_view = 0
        app.main.destroy()
    app.createWidgets()

print(app.views)
# Debug commands
root.bind('<F4>', end_it)
root.bind('<F5>', switch_view)
root.focus_force()
root.mainloop()
