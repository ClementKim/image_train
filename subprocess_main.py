import os
import sys
import subprocess

from openpyxl import Workbook

if not "result" in os.listdir(os.getcwd()):
    os.mkdir("result")

rst_dir = os.path.join(os.getcwd(), "result")

if not "result.xlsx" in os.listdir(rst_dir):
    write_wb = Workbook()
    write_ws = write_wb.active
    write_ws['A1'] = 'loss function'
    write_ws['B1'] = 'optimizer'
    write_ws['C1'] = 'learning rate'
    write_ws['D1'] = 'epochs'
    write_ws['E1'] = 'avg. accuracy'

    write_wb.save(os.path.join(rst_dir, "result.xlsx"))

loss_fn_list = ["cross"]

learning_rate_list = [1e-5]
epoch_list = [10, 50, 100]
optimizer_list = ["sgd"]

lst = []
idx = 1
for optimizer in optimizer_list:
    for loss_fn in loss_fn_list:
        for learning_rate in learning_rate_list:
            for epochs in epoch_list:
                for _ in range(10):
                    ipt = str(optimizer) + "_" + str(learning_rate) + "_" + loss_fn + "_" + str(epochs) + "_" + str(idx)
                    output = subprocess.Popen([sys.executable, "-u", "main.py", ipt, "-r"], stdout = subprocess.PIPE, stderr = subprocess.STDOUT, universal_newlines = True)
                
                    while output.poll() == None:
                        out = output.stdout.readline()
                        print(out, end = '')

                    idx += 1

print("All tests are Done")

