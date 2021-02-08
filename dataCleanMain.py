from nltk.sentiment.vader import SentimentIntensityAnalyzer
import json
import csv
import re

""" partially used some or similar code from from sources below:
source [1]: https://www.geeksforgeeks.org/python-convert-a-list-to-dictionary/
source [2]: https://stackoverflow.com/questions/5788521/reading-a-csv-file-using-python
source [3]: https://www.geeksforgeeks.org/convert-json-to-dictionary-in-python/
source [4]: https://thispointer.com/python-how-to-sort-a-dictionary-by-key-or-value/"""

# 1
def download1(text):
    f = open(text)
    comp = dict()

    for line in f:
        line = line.replace("Name:", "")
        line = line.strip("''")
        line = line.strip()
        line = line.split(",Purpose:")
        for word in range(len(line) - 1):
            term1 = line[word]
            term2 = line[word + 1]
            sid = SentimentIntensityAnalyzer()
            ss = sid.polarity_scores(term2)
            comp[term1] = ss["compound"]
    f.close()
    print(len(comp))
    return comp


# 2
def download2(text):
    with open(text) as json_file:
        data = json.load(json_file)
        comp1 = dict()
    for i in data:
        term1 = i["name"]
        term2 = i["purpose"]
        sid = SentimentIntensityAnalyzer()
        ss = sid.polarity_scores(term2)
        comp1[term1] = ss["compound"]
    json_file.close()
    print(len(comp1))
    return comp1


#  source (1)
def convert_list(lst):
    res_dct = {lst[i]: lst[i + 1] for i in range(0, len(lst), 2)}
    return res_dct


# 3
def download3(text):
    f = open(text)
    lst = []
    for line in f:
        line = line.strip()
        line = line.replace("Name:", "")
        line = line.replace("Company Details", "")
        line = line.replace("Purpose:", "")
        line = line.strip()
        lst.append(str(line))
    res_dct = {lst[i]: lst[i + 1] for i in range(1, len(lst), 2)}

    for key in res_dct:
        sid = SentimentIntensityAnalyzer()
        res_dct[key] = sid.polarity_scores(res_dct[key])["compound"]
    f.close()
    print(len(res_dct))
    return res_dct


# 4 source[2]
def download4(file_csv):
    with open(file_csv, newline="") as f:
        reader = csv.reader(f)
        data = dict(reader)
        del data["Name"]
    for key in data:
        sid = SentimentIntensityAnalyzer()
        data[key] = sid.polarity_scores(data[key])["compound"]
    f.close()
    print(len(data))
    return data


# 5
def download5(file):
    with open(file, newline="") as f:
        reader = csv.reader(f)
        data = dict(reader)
        del data["name"]
        for key in data:
            sid = SentimentIntensityAnalyzer()
            data[key] = sid.polarity_scores(data[key])["compound"]
    print(len(data))
    return data


# 6 + 7
def download6(file):
    with open(file, newline="") as f:
        reader = csv.reader(f)
        dict1 = {}
        for line in reader:
            dict1[line[1]] = line[2]
    del dict1["Name"]
    for key in dict1:
        sid = SentimentIntensityAnalyzer()
        dict1[key] = sid.polarity_scores(dict1[key])["compound"]
    f.close()
    print(len(dict1))
    return dict1


# 8 source[3]
def download8(file):
    with open(file) as json_file:
        data = json.load(json_file)
    comp = dict(data)
    for key in comp:
        sid = SentimentIntensityAnalyzer()
        comp[key] = sid.polarity_scores(comp[key])["compound"]
    json_file.close()
    print(len(comp))
    return comp


# 9
def download9(text):
    list1 = []
    compDict = dict()
    f = open(text, "r")
    for line in f:
        line = line.replace("Name:", "")
        line = line.strip()
        list1.append(line)
    for item in range(len(list1) - 50):
        compName = list1[item]
        compPurpose = list1[item + 50]
        compDict[compName] = compPurpose
        sid = SentimentIntensityAnalyzer()
        ss = sid.polarity_scores(compPurpose)
        compDict[compName] = ss["compound"]
    print(len(compDict))
    return compDict


# 10
def download10(text):
    dictComp = dict()
    with open(text, "rt") as text_file:
        contents = text_file.read()
        names = re.findall("Name: ([a-zA-Z\s,]+)", contents)
        purpose = re.findall("Purpose: ([a-zA-Z\s,-]+)", contents)
        for i in range(0, len(names)):
            dictComp[names[i]] = purpose[i]
        for key in dictComp:
            sid = SentimentIntensityAnalyzer()
            dictComp[key] = sid.polarity_scores(dictComp[key])["compound"]
    print(len(dictComp))
    return dictComp


def mergeDict(dict1, dict2, dict3, dict4, dict5, dict6, dict7, dict8, dict9, dict10):
    # Step 1; merge dictionaries
    dict_merged = {
        **dict1,
        **dict2,
        **dict3,
        **dict4,
        **dict5,
        **dict6,
        **dict7,
        **dict8,
        **dict9,
        **dict10,
    }
    # Step2: sort key,values : source[4]
    listofTuples = sorted(dict_merged.items(), key=lambda x: x[1])

    # Step3:  print top 10 and bottom 10
    print(":::::::::::::::list::::::::::::::")
    for elem in listofTuples:
        print(elem[0], ",", elem[1])
    # Step4:  print top 10 and bottom 10
    print(":::::::::::::::bottom ten::::::::::::::")
    for elem in listofTuples[0:10]:
        print(elem[0], ",", elem[1])
    print(":::::::::::::::top ten:::::::::::")
    for elem in listofTuples[-11:-1]:
        print(elem[0], ",", elem[1])
    print(len(listofTuples))


if __name__ == "__main__":
    print(download1("download1.txt"))
    print(download2("download7.json"))
    print(download3("download4.txt"))
    print(download4("download8.csv"))
    print(download5("download10.csv"))
    print(download6("download11.csv"))
    print(download6("download13.csv"))
    print(download8("download12.txt"))
    print(download9("download5.txt"))
    print(download10("download3.txt"))
    print(
        mergeDict(
            download1("download1.txt"),
            download2("download7.json"),
            download3("download4.txt"),
            download4("download8.csv"),
            download5("download10.csv"),
            download6("download11.csv"),
            download6("download13.csv"),
            download8("download12.txt"),
            download9("download5.txt"),
            download10("download3.txt"),
        )
    )
