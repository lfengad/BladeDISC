import sys, os

def exec():
    with open(sys.argv[1] + "/tune.log") as f:
        lines = f.readlines()
    dic = {}
    for i in lines:
        items = i.split(":")
        dic[items[0].strip()] = items[1].strip()

    new_dic = sorted(dic.items(), key=lambda item: float(item[1]))
    print(new_dic)
    if not os.path.exists(sys.argv[1] + "tune_sort.log"):
        with open(sys.argv[1] + "/tune_sort.log", "a") as f:
            for name, t in new_dic:
                f.write(name + " : " + t + "\n")

if __name__ == "__main__":
    exec()
