import os, sys

lambda2 = [5, 50, 1000, 5000]
alpha = [0.25, 0.5, 0.75]
rho = [5000, 10000, 15000]

for i in range(4):
    for j in range(3):
        for k in range(3):
            terminal_command = "python train.py --config_files ADMMSLIM --model ADMMSLIM --lambda2 " + str(lambda2[i]) + " --alpha " + str(alpha[j]) + " --rho " + str(rho[k])
            print(terminal_command)
            os.system(terminal_command)
 
            
    