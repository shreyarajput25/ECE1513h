import csv
import numpy as np

with open('bank-full.csv', newline='') as inp:
    csv_reader = csv.reader(inp, delimiter=';')
    line_count = 0
    Employemnt = ["services", "unknown", "unemployed", "entrepreneur", "blue-collar", "management","admin.", "technician", "self-employed", "housemaid", "retired", "student", ]
    employment_replace = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"] 
    martial = ["married", "single", "divorced"]
    martial_new = ["1", "2", "3"]
    education = ["unknown", "secondary", "primary", "tertiary"]
    education_new = ["1", "2", "3", "4"]
    binary = ["yes","no"]
    binary_new = ["1", "0"]
    data = []
    for row in csv_reader:
            #print(row[1])
            
            for i in range(len(Employemnt)):
                row[1] = row[1].replace(Employemnt[i], employment_replace[i])
            for i in range(len(martial)):
                row[2] = row[2].replace(martial[i], martial_new[i])
            for i in range(len(education)):
                row[3] = row[3].replace(education[i], education_new[i])
            for i in range(len(binary)):
                row[4] = row[4].replace(binary[i], binary_new[i])
                row[6] = row[6].replace(binary[i], binary_new[i])
                row[7] = row[7].replace(binary[i], binary_new[i])
                row[16] = row[16].replace(binary[i], binary_new[i])
            for i in range(len(row[8])):
                for j in range(8, 12):
                    row[j] = row[j+4]
            row[11] = row[16]
            del row[16]
            del row[15]
            del row[14]
            del row[13]
            del row[12]
            if line_count != 0:
                data.append(row)
            line_count = 1+ line_count

full_data  = np.zeros((45211, 28))     
job = np.zeros((45211, 19), dtype=int)
married = np.zeros((45211, 3), dtype=int)
edu = np.zeros((45211, 4), dtype=int)
for i in range(1, 45211):
    a = data[i][1]
    a =int(a)-1
    b = data[i][2]
    b =int(b)-1
    c = data[i][3]
    c =int(c)-1
    job[i][a] = 1
    job[i][12+b] =1
    job[i][15+c] =1
    #married[i][b] = 1
    #edu[i][c] = 1

for j in range(0,45211):
    full_data[j][0]= data[j][0]
    for i in range(1,8):
        full_data[j][i]= data[j][i+3]
    for i in range(8,27):
        full_data[j][i] = job[j][i-8]
    full_data[j][27] = data[j][11]

with open('bank_final_last.csv', mode='w', newline='') as out:
    
    fwrite = csv.writer(out, delimiter=';')
    for i in range(0,45211):
        fwrite.writerow(full_data[i][:])
             