from datasets import load_dataset

def main():
    data = (load_dataset('json', data_files='source/train.json',split='train')).train_test_split(test_size=0.2)
    print(data['train'][0])





if __name__ == '__main__':
    main()
