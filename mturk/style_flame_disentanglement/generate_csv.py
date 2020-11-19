import sys
sys.path.append('../../')
import constants as cnst
import pandas
import os

root_url = f'{cnst.amt_bucket_base_url}/textured_rendering/'
image_dir = f'{cnst.output_root}sample/29/flame_param_association_eval/textured_rendering/'
image_names = os.listdir(image_dir)
for i, file_name in enumerate(image_names):
    image_names[i] = root_url + file_name

df = pandas.DataFrame(data={"image_url": image_names})
csv_path = f'{cnst.output_root}sample/29/flame_param_association_eval/'
df.to_csv(os.path.join(csv_path, "flm_asso_10k.csv"), sep=',', index=False)