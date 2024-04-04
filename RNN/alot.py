import os


for i in range(1000):
    try:
        os.system("python3 train.py")
    except:
        print("")
