
from typing import List

def read(filepath:str) -> List[List[str]]:

    f = open(filepath, 'r', encoding='utf-8')

    all_instances = []
    inst = []
    for line in f.readlines():
        line = line.rstrip()
        if line == "":
            all_instances.append(inst)
            inst = []
            continue
        inst.append(line)

    f.close()
    return all_instances

def write(insts: List[List[str]], outfile:str):
    f = open(outfile, 'w', encoding='utf-8')

    for inst in insts:
        for line in inst:
            f.write(line + '\n')
        f.write('\n')

    f.close()


# insts = read("./datasets/ontonotes_chinese/train.sd.conllx")
# write(insts[34800:], "./fixtures/sample_large_chinese.conllx")