import os
import csv


def csv_writer(path, label, wr):
    files = os.listdir(path + "/" + label)

    for f in files:
        row = []
        data = open(path + "/" + label + "/" + f).read()
        data = data.replace('\n', ' ').replace('\r', '')
        row.append(data)
        row.append(label)
        wr.writerow(row)


p = "/home/oee/Desktop/category"

dirs = os.listdir(p)

myfile = open("/home/oee/Desktop/ml_csv/all_data.csv", 'wb')
wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
header = ['text', 'label']
wr.writerow(header)
for d in dirs:
    if "." not in d:
        continue
    name = d.split(".")[0]
    print name
    csv_writer(p, name, wr)
myfile.close()

# label = "accident"
# path = "/home/oee/Desktop/category/" + label

# files = os.listdir(path)
# myfile = open("/home/oee/Desktop/ml_csv/all_data.csv", 'wb')
# wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
# header = ['text', 'label']
# wr.writerow(header)
# for f in files:
#     row = []
#     data = open(path + "/" + f).read()
#     data = data.replace('\n', ' ').replace('\r', '')
#     row.append(data)
#     row.append(label)
#     wr.writerow(row)
# myfile.close()
# data = open("/home/oee/Desktop/category/accident/link1.txt").read()
