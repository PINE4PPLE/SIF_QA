import pandas as pd 
import re
def read_data(file_path):
    data=pd.read_excel(file_path)
    new_data=data.iloc[:, 0:2]
    new_data.columns=[["Question", "Answers"]]
    new_data=new_data[:-1]
    return new_data

origin = read_data("../DATA/all.xlsx")

year = "[0-9][0-9][0-9][0-9]"
y = [re.search(year, i[0]) != None for i in origin["Question"].values]
other = [not i for i in y]
df_y = origin[y]
df_other = origin[other]
df_y.to_csv ('../DATA/year_related.csv',index=False)
df_other.to_csv ('../DATA/other.csv',index=False)