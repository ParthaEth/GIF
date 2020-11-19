import sys
sys.path.append('../../')
import constants as cnst
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

font = {'family' : 'Nimbus Roman',
	    # 'sans-serif':'Nimbus Roman',
        'size'   : 15}
matplotlib.rc('font', **font)

#df = pd.read_csv('./textured_rend_flm_asso_right_likert_scale.csv')
df = pd.read_csv(cnst.flame_style_vec_association_result_csv_path)
results = df[['Input.image_url', 'Answer.category.label']]
#my_table = df[df['model'].str.match('Mac')]

#['1 aspect similar', '2 aspects similar', '3 aspects loosely similar', 'All aspects highly similar']
# 'Strongly disagree', 'Disagree', 'Neither agree nor disagree', 'Agree', 'Strongly agree'
results = results.replace('Strongly disagree',1.0)
results = results.replace('Disagree',2.0)
results = results.replace('Neither agree nor disagree',3.0)
results = results.replace('Agree',4.0)
results = results.replace('Strongly agree',5.0)


#print(results)
my_hist = []
for i in range(0,10):
	#print(str(i))
	person = results[results['Input.image_url'].str.contains(f'rendering/{str(i)}_')]
	#mode = person['Answer.category.label'].mode()
	median = person['Answer.category.label'].median()
	#print(person)	
	print(f'{i} --> {median}')
	my_hist.append(person['Answer.category.label'])
	#ax = person['Answer.category.label'].plot.hist(bins=[0.5,1.5,2.5,3.5,4.5,5.5], alpha = .5)
	#ax.plot()
plt.hist(my_hist,bins=[0.5,1.5,2.5,3.5,4.5,5.5],alpha=.5)
plt.grid(True)
plt.xlabel('5-Point Likert Scale')
plt.ylabel('Rating Frequency')
plt.savefig('./hist_10_styles_style_flame_independence.pdf', bbox_inches="tight")
plt.show()


