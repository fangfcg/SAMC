# -*- coding: utf-8 -*-
import os
import csv

DATA_ROOT = "./"

class DataLoader(object):
    def __init__(self):
        pass

    def get_data(self, path):
        with open(path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            rows = [row for row in reader]
        return rows

    def write_to_file(self, data, filepath, split=","):
        newStr = ''
        result = data
        for row in result:
            for t in row[:-1]:
                newStr += str(t)
                newStr += split
            newStr += str(row[-1])
            newStr += '\n'

        with open(filepath, "w") as outputFile:
            outputFile.write(newStr)
            outputFile.close()


