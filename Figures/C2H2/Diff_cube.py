#!/usr/bin/python

# print a file 'output.cube'
# containing the difference between two cube files

import sys

initial_line = 14

file1 = open(sys.argv[-2])
file2 = open(sys.argv[-3])
file3 = sys.argv[-1]
file_out = open(file3, 'w')

f1 = file1.readlines()
f2 = file2.readlines()

string_out = ''
for i in range(initial_line):
  string_out += f1[i]
for i in range(initial_line,len(f1)):
  line1 = f1[i].split()
  line2 = f2[i].split()
  for j in range(len(line1)):
    string_out += str(float(line1[j])-float(line2[j]))
    string_out += ' '
  string_out += ("\n") 

print >> file_out, string_out

