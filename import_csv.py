import csv

def importCSV(path, x, y):
    f = open(path, 'r')
    reader = csv.reader(f)
    next(reader)

    for row in reader:
        theta_in = float(row[0])
        phi_in = float(row[1])
        theta_out = float(row[2])
        phi_out = float(row[3])
        red = float(row[4])
        green = float(row[5])
        blue = float(row[6])

        x.append([theta_in, phi_in, theta_out, phi_out])
        y.append([red, green, blue])
        