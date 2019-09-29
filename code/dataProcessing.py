from collections import defaultdict, OrderedDict

import math
import numpy as np

# user - age
ua_dict = defaultdict(int)
# user - occupation
uo_dict = defaultdict(int)
# movie - type(genre)
mt_dict = defaultdict(list)

prefix = "../data/"

u_info = prefix + "u.user"
u_occupation = prefix + "u.occupation"
u_item = prefix +"u.item"
u_genre = prefix + "u.genre"
u_inter = prefix + "u.data"

# outputfile_name
ua_out = prefix + "ml-100k.ua"
uo_out = prefix + "ml-100k.uo"
mt_out = prefix + "ml-100k.mt"
uu_knn = prefix + "ml-100k.uu_knn_50"
mm_knn = prefix + "ml-100k.mm_knn_50"


#prefix_outputfile = ""

def user_age_construction():
    with open(u_info, 'r') as infile:
        for line in infile:
            user_id, age = line.split("|")[0:2]
            age = int(age)
            if age % 10 == 0:
                ua_dict[user_id] = math.floor(age / 10)
            else:
                ua_dict[user_id] = math.floor((age / 10)) + 1
    with open(ua_out, "w") as outfile:
        for user, age in ua_dict.items():
            outfile.write(str(user) + '\t' + str(age) + '\n')

def user_occupation_construction():
    occupation2idx = defaultdict(int)
    idx2occupation = defaultdict(str)

    idx = 1
    with open(u_info, "r") as infile:
        for line in infile:
            occ = line.split("|")[3]
            if occ not in occupation2idx:
                occupation2idx[occ], idx2occupation[idx] = idx, occ
                idx += 1

    with open(u_info, "r") as infile:
        for line in infile:
            user_id, occupation = line.split("|")[0], line.split("|")[3]
            occ_idx = occupation2idx[occupation]
            uo_dict[user_id] = occ_idx

    with open(uo_out, "w") as outfile:
        for user, occupation in uo_dict.items():
            outfile.write(str(user) + '\t' + str(occupation) + '\n')

def movie_genre_construction():

    with open(u_item, "r", encoding='utf-8') as infile:
        for line in infile:
            line = line.strip().split("|")
            movie_id, genre_list = line[0], np.array([int(i) for i in line[-19:]])
            mt_dict[movie_id] = np.nonzero(genre_list)[0]

    with open(mt_out, "w") as outfile:
        for user, genre_list in mt_dict.items():
            for genre in genre_list:
                outfile.write(str(user) + '\t' + str(genre) + '\n')

def knn_construction():
    # user -> item; item -> user
    ui_dict = defaultdict(list)
    iu_dict = defaultdict(list)

    uu_dict = defaultdict(list)
    ii_dict = defaultdict(list)

    uu_score_dict = defaultdict(dict)
    ii_score_dict = defaultdict(dict)

    with open(u_inter, "r") as infile:
        for line in infile:
            user_id, item_id = line.split()[0:2]
            user_id, item_id = int(user_id), int(item_id)
            ui_dict[user_id].append(item_id)
            iu_dict[item_id].append(user_id)

    for user_id, item_list in ui_dict.items():
        for item in item_list:
            uu_dict[user_id] += iu_dict[item]
        uu_dict[user_id] = list(set(uu_dict[user_id]))

    for item_id, user_list in iu_dict.items():
        for user in user_list:
            ii_dict[item_id] += ui_dict[user]
        ii_dict[item_id] = list(set(ii_dict[item_id]))

    for user_id, user_list in uu_dict.items():
        u1_items = ui_dict[user_id]
        u1_num = len(u1_items)
        for user2_id in user_list:
            u2_items = ui_dict[user2_id]
            u2_num = len(u2_items)

            intersection_num = len([u for u in u1_items if u in u2_items])

            uu_score_dict[user_id][(user_id, user2_id)] = intersection_num / np.sqrt(u1_num * u2_num)

    for item_id, item_list in ii_dict.items():
        i1_items = iu_dict[item_id]
        i1_num = len(i1_items)
        for item2_id in item_list:
            i2_items = iu_dict[item2_id]
            i2_num = len(i2_items)

            intersection_num = len([i for i in i1_items if i in i2_items])

            ii_score_dict[item_id][(item_id, item2_id)] = intersection_num / np.sqrt(i1_num * i2_num)

    with open(uu_knn, "w") as outfile:
        uu_score_dict = OrderedDict(sorted(uu_score_dict.items(), key=lambda x: x[0]))
        for user_id, _ in uu_score_dict.items():
            sorted_uu = sorted(uu_score_dict[user_id].items(), key=lambda x: x[1], reverse=True)[1:51]
            for uu2val in sorted_uu:
                u1, u2, val = uu2val[0][0], uu2val[0][1], uu2val[1]
                outfile.write(str(u1) + "\t" + str(u2) + '\t' + str(val) + "\n")

    with open(mm_knn, "w") as outfile:
        ii_score_dict = OrderedDict(sorted(ii_score_dict.items(), key=lambda x: x[0]))
        for item_id, _ in ii_score_dict.items():
            sorted_ii = sorted(ii_score_dict[item_id].items(), key=lambda x: x[1], reverse=True)[1:51]
            for ii2val in sorted_ii:
                i1, i2, val = ii2val[0][0], ii2val[0][1], ii2val[1]
                outfile.write(str(i1) + "\t" + str(i2) + '\t' + str(val) + "\n")

if __name__ == '__main__':

    user_age_construction()
    user_occupation_construction()
    movie_genre_construction()
    knn_construction()
