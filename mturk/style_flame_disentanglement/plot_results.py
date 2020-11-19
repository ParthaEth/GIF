import sys
sys.path.append('../../')
import constants as cnst
import pandas
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np


df = pandas.read_csv(cnst.five_pt_likert_scale_result_csv_path)

results = df[['Input.image_url', 'Answer.category.label']].to_numpy()
print(df)
id_hist_dict = {}


categories = ['Strongly disagree', 'Disagree', 'Neither agree nor disagree', 'Agree', 'Strongly agree']
categories_dict = {}
for i, cat in enumerate(categories):
    categories_dict[cat] = i

fontsize = 15

for i in range(0, len(results)):
    id = int(results[i, 0].split('/')[-1].split('_')[0])
    # import ipdb; ipdb.set_trace()
    if id not in id_hist_dict:
        id_hist_dict[id] = {}

    if results[i, -1] in id_hist_dict[id]:
        id_hist_dict[id][results[i, -1]] += 1
    else:
        id_hist_dict[id][results[i, -1]] = 1

ids = []
scores = []
for key in id_hist_dict.keys():
    # import ipdb; ipdb.set_trace()
    current_dict = id_hist_dict[key]
    user_ratings = []
    total_imaged_per_id = 0
    for i, key_cat in enumerate(categories):
        if key_cat in current_dict:
            user_ratings.extend(current_dict[key_cat] * [(i + 1), ])
            total_imaged_per_id += current_dict[key_cat]

    user_ratings = np.bincount(user_ratings).argmax()
    # print(total_imaged_per_id)
    print(f'{key}: {user_ratings} from {total_imaged_per_id} samples')
    ids.append(key)
    scores.append(user_ratings)

print(f'Total_mean = {np.mean(scores)}')

plt.bar(ids, scores, align='center', alpha=0.5)
plt.xticks(ids, ids)
plt.ylim(bottom=1, top=5)
plt.ylabel('User Scores', fontsize=fontsize)
plt.xlabel('Style ID', fontsize=fontsize)
# plt.title('Programming language usage')

plt.savefig('bar_graph_style_disentanglement.pdf', format='pdf', bbox_inches='tight')

plt.figure()
# Draw histogram of all
ratings = []
for str_rating in results[:, -1]:
    # import ipdb; ipdb.set_trace()
    ratings.append(categories_dict[str_rating] + 1)

ratings = np.array(ratings)
# import ipdb; ipdb.set_trace()
plt.hist(ratings, bins=np.arange(ratings.min(), ratings.max()+2) - 0.5)
plt.savefig('rating_hist.png')

