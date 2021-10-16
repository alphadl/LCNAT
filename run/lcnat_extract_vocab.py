import sys

with open(sys.argv[1], 'r', encoding='utf-8') as fin:
    with open(sys.argv[2], 'w', encoding='utf-8') as fout:
        for line in fin:
            idx = line.rfind(' ')
            word = line[:idx]
            fout.write(word+"\n")