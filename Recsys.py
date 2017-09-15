from User import User
from Item import Item
import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd
import random


class Recsys:
    """
    Recommender system class

    Attributes:
        users: dict, key -> user id, value -> User object
        items: dict, key -> item id, value -> Item object
        lr: LogisticRegression object
        X_train: Dataframe of recommended items for training model. Format: (m, n). m samples with n features.
        Y_train: Dataframe of recommended items for training model. Format: (m, 1). m samples with labels.
    """

    def __init__(self):
        """Assign initial values to attributes"""
        self.users = dict()
        self.items = dict()
        self.lr = LogisticRegression()
        self.X_train = pd.DataFrame()
        self.Y_train = pd.DataFrame()

    def add_user(self, u):
        """
        Add a user to system
        :param u: User object
        :return: None
        """
        if u.id not in self.users.keys():
            self.users[u.id] = u

    def add_item(self, i):
        """
        Add an item to system
        :param i: Item object
        :return: None
        """
        if i.id not in self.items.keys():
            self.items[i.id] = i

    def get_relevance(self, v1, v2):
        """
        Calculate the relevance between v1 and v2. Default method is cosine similarity.
        :param v1: array
        :param v2: array
        :return: float
        """
        rel = np.sum(v1 * v2) / np.sqrt(np.sum(np.square(v1)) * np.sum(np.square(v2)))
        return rel

    def recommend_preference_content(self, uid):
        """
        Recommend items for a user by calculating relevance between user's preference and item's content.
        :param uid: string
        :return: a list of recommended items
        """
        user = self.users[uid]
        preference = user.preference
        rec_list = list()
        for iid, item in self.items.items():
            rel = self.get_relevance(preference, item.topic)
            rec_list.append((iid, rel))
        rec_list = sorted(rec_list, key=lambda x: x[1], reverse=True)
        return rec_list

    def recommend_title_rel(self, uid):
        """
        Recommend items for a user by calculating relevance between items' title and that of user's click record
        :param uid: string
        :return: a list of recommended items
        """
        user = self.users[uid]
        click_record = user.click_record
        rec_list = list()
        for click_iid in click_record:
            for iid, item in self.items.items():
                if iid != click_iid:
                    click_item = self.items[click_iid]
                    rel = self.get_relevance(click_item.title, item.title)
                    rec_list.append((iid, rel))
        rec_list = sorted(rec_list, key=lambda x: x[1], reverse=True)
        return rec_list

    def extract_features(self, uid, candidate_list):
        """
        Extract features for sort. Currently using simple manual methods.
        :param uid: string
        :param candidate_list: a list of recommended items
        :return: a Dataframe of candidate items for sort. Format: (m, n). m samples with n features.
        """
        df = pd.DataFrame(candidate_list, columns=['iid', 'recommend_value'])
        for i in range(len(df)):
            iid = df.loc[i, 'iid']
            df.loc[i, 'click_sum'] = self.items[iid].click_sum

        self.X_train = self.X_train.append(df)
        return df

    def train_sort_model(self):
        """
        Train sorting model using history click records.
        :return: None
        """
        X_columns = list(self.X_train.columns)
        X_columns.remove('iid')
        X_train = self.X_train[X_columns]

        Y_train = self.Y_train['label']
        # To guarantee there exists 2 classes, modify some fake data
        if 0 not in Y_train.values:
            Y_train.loc[0, 'label'] = 0
        if 1 not in Y_train.values:
            Y_train.loc[0, 'label'] = 1

        self.lr.fit(X_train.values, Y_train.values)

    def ml_sort(self, uid, featured_candidate_df):
        """
        Sort the candidate items using machine learning methods.
        :param uid: string
        :param featured_candidate_df: a df of candidate items for sort. Format: (m, n). m samples with n features.
        :return: a list of sorted recommended items. Format: (iid, prob).
        """
        # Predict CTR using LR model (currently global level)
        cols = list(featured_candidate_df.columns)
        cols.remove('iid')
        predicted_result = self.lr.predict_proba(featured_candidate_df[cols].values)
        predicted_result = list(predicted_result[:, 1])

        sorted_list = list()
        for i in range(len(featured_candidate_df)):
            sorted_list.append((featured_candidate_df.loc[i, 'iid'], predicted_result[i]))
        sorted_list = sorted(sorted_list, key=lambda x: x[1], reverse=True)

        return sorted_list

    def recommend_for(self, uid, N):
        """
        Recommend N items for a user. Use >1 algorithms, mix the results, filter by some rules and sort them.
        :param uid: string
        :return: a list of recommended items
        """
        if uid not in self.users.keys():
            return None

        candidate_list = list()
        algorithms = list()
        weights = list()
        mixed_candidate_list = list()

        # Multiple algorithms
        algorithms.append(self.recommend_preference_content)
        algorithms.append(self.recommend_title_rel)

        for algo in algorithms:
            candidate_list.append(algo(uid))

        # Assign weights
        count = len(algorithms)
        for i in range(count):
            weights.append(1.0 / (count * 1.0))

        # Mix results
        for i in range(count):
            n = int(N * weights[i])
            mixed_candidate_list += candidate_list[i][:n]

        # Merge duplicated items and filter out improper items
        tmp_dic = dict()
        for iid, val in mixed_candidate_list:
            if iid not in tmp_dic.keys():
                tmp_dic[iid] = 0
            tmp_dic[iid] += val
        mixed_candidate_list.clear()
        for iid, val in tmp_dic.items():
            if iid not in self.users[uid].dislike_set and iid not in self.users[uid].click_record:
                mixed_candidate_list.append((iid, val))

        if len(mixed_candidate_list) == 0:
            return []

        # Sort items
        if len(self.X_train) > 0 and len(self.Y_train) > 0:
            featured_candidate_df = self.extract_features(uid, mixed_candidate_list)
            final_rec_list = self.ml_sort(uid, featured_candidate_df)
        else:
            final_rec_list = sorted(mixed_candidate_list, key=lambda x: x[1], reverse=True)
            self.extract_features(uid, final_rec_list)

        return final_rec_list

    def get_feedback(self, uid, rec_list):
        """
        Get a user's feedback to the recommended items list. Currently use fake random data.
        :param uid: string
        :param rec_list: a list of recommended items
        :return: a list of tuples denoting whether items are clicked. Format: (iid, 1), 1 for clicked, 0 for not clicked
        """
        feedback = list()
        for iid, prob in rec_list:
            feedback.append((iid, random.randint(0, 1)))

        return feedback

    def deal_with_feedback(self, uid, feedback):
        """
        Deal with a user's feedback to the recommended items. Adjust user's preference, click record. Adjust item's
        click_sum. Update lr model.
        :param uid: string
        :param feedback: a list of tuples denoting whether items are clicked
                         Format: (iid, 1), 1 for clicked, 0 for not clicked
        :return: None
        """
        n = 5
        evaluation = self.get_evaluation(feedback, n)
        print(evaluation)

        for i in range(len(feedback)):
            iid = feedback[i][0]
            clicked = feedback[i][1]
            if clicked == 1:
                self.users[uid].click_record.add(iid)
                self.users[uid].preference = (self.users[uid].preference + self.items[iid].topic) \
                    / len(self.users[uid].click_record)
                self.items[iid].click_sum += 1

        df = pd.DataFrame(feedback, columns=['iid', 'label'])
        self.Y_train = self.Y_train.append(df)
        self.train_sort_model()

    def get_evaluation(self, feedback, n):
        """
        Get the evaluation of feedback. Currently using average precision.
        :param feedback: a list of tuples denoting whether items are clicked
                         Format: (iid, 1), 1 for clicked, 0 for not clicked
               n: int, position parameter
        :return: dict
        """
        evaluation = dict()

        count = 0
        precision = 0.0
        for i in range(len(feedback)):
            if feedback[i][1] == 1:
                count += 1
                if i < n:
                    precision += count * 1.0 / (i + 1)
        if count != 0:
            precision /= count
        evaluation['average precision'] = precision

        return evaluation
