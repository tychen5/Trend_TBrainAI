import csv, pickle

def readCSV(file):
    # return 2d numpy array of string
    data = []
    f = open(file, 'r')
    for row in csv.reader(f):
        data.append(row)
    f.close()
    return data


def readPickle():

    
def writePickle():