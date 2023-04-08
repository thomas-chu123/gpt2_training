import json, os, sys, csv

def main():
    testplan = []
    # create a function to read the source/testplan.csv and convert all line to dict data
    with open("source/testplan.csv", encoding="utf-8") as file:
        text = csv.reader(file)
        header = next(text)
    # text_list = text.split("\n")
        for case in text:
            testcase = {"case": ''}
            testcase["case"] = case[0] + "\n" + case[1] + "\n" + case[2]
            testplan.append(testcase)
            print(testcase["case"])

    with open("source/train_tp.json","w",encoding="utf-8") as f:
        f.write(json.dumps(testplan))

if __name__ == "__main__":
    main()