import re
with open("./sentences_man.txt","r") as fman:
    with open("./sentences_woman.txt","w") as fwoman:
        for line in fman:
            # print(line)
            text = re.sub(r'man',"woman",line)
            text = re.sub(r"boy","girl",text)
            text = re.sub(r"grandpa","grandma",text)
            text = re.sub(r"Grandpa","Grandma",text)
            text = re.sub(r"grandfather","grandmother",text)
            print(text)
            fwoman.write(text)
            