import pandas as pd 
path = "C:/Users/leo/Desktop/wo_qa/knowledgequiz"
f = open(path + "/DATA/Question.txt", 'w', encoding='utf-8')
df = pd.read_csv(path + "/DATA/clean.csv")
for q in df["Question"]:
    f.write(q)
f.close()
