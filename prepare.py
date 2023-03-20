import json, os, sys

def main():
    script_list = []
    start_flag = False
    end_flag = False
    script_string = {}
    count = 0

    with open("source/Function_V6.2.dlf","r",encoding="utf-8") as f:
        text = f.read()
    text_list = text.split("\n")
    script_string = {'case_name': '', 'case_script': ''}
    for text_line in text_list:
        if text_line.startswith("# "):
            # for next new script line
            if start_flag == True and end_flag == False:
                start_flag = True
                end_flag = False
                script_list.append(script_string)
                count = count + 1
                print(count, script_string)
                script_string = {'case_name': '', 'case_script': ''}
                script_string["case_name"] = text_line.replace('# ', '')
            # for new start test item
            elif start_flag == False and end_flag == False:
                start_flag = True
                end_flag = False
                script_string = {'case_name': '', 'case_script': ''}
                script_string["case_name"] = text_line.replace('# ', '')
        else:
            if start_flag==True and end_flag==False:
                script_string["case_script"] = script_string["case_script"] + text_line + "\r\n"

    with open("source/train.json","w",encoding="utf-8") as f:
        f.write(json.dumps(script_list))

if __name__ == "__main__":
    main()