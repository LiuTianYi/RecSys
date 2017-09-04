from User import User
from Item import Item
from Recsys import Recsys
import random
import numpy as np

CONTENT_DIM = 100
TITLE_DIM = 50
USERS = 100
ITEMS = 100
N = 10
ITERATIONS = 10

if __name__ == '__main__':
    recsys = Recsys()

    for i in range(USERS):
        uid = 'u_' + str(i)
        gender = random.randint(0, 1)
        age = random.randint(1, 100)
        preference = np.random.rand(1, CONTENT_DIM)[0]
        user = User(uid, gender, age, preference)
        recsys.add_user(user)

    for i in range(ITEMS):
        iid = 'i_' + str(i)
        topic = np.random.rand(1, CONTENT_DIM)[0]
        title = np.random.rand(1, TITLE_DIM)[0]
        item = Item(iid, topic, title)
        recsys.add_item(item)

    for i in range(ITERATIONS):
        for uid in recsys.users.keys():
            rec_list = recsys.recommend_for(uid, N)
            print(rec_list)
            if len(rec_list) == 0:
                continue
            feedback = recsys.get_feedback(uid, rec_list)
            recsys.deal_with_feedback(uid, feedback)

    print(recsys.X_train.describe())
    print(recsys.Y_train.describe())