from pathlib import Path


def atleastk_to_atmostk(path):
    assert Path(path).suffix == ".cnfcc"
    path_res = path[:-len(".cnfcc")] + "_atmostk.cnfcc"

    file = open(path, 'r')
    file_res = open(path_res, 'w')
    while True:
        line = file.readline()
        if not line:
            break

        if line[0] == 'c' or line[0] == 'p':
            file_res.write(line)
            continue

        info = line.split()
        if info[-2] != '#':
            file_res.write(line)
            continue

        n = len(info) - 4
        if n - int(info[-3]) < 0:
            file_res.write(str(-int(info[-1])) + ' 0\n')
            continue

        for i in info:
            if i == ">=":
                break
            file_res.write(str(-int(i)) + ' ')
        file_res.write('<= ' + str(n - int(info[-3])) + ' # ' + info[-1] + '\n')
    file.close()
    file_res.close()


atleastk_to_atmostk("../../data/CNFs/universal_adversarial_robustness_model0.cnfcc")
atleastk_to_atmostk("../../data/CNFs/universal_adversarial_robustness_model4.cnfcc")
atleastk_to_atmostk("../../data/CNFs/universal_adversarial_robustness_model5.cnfcc")
atleastk_to_atmostk("../../data/CNFs/equality_model0_and_model0.cnfcc")
atleastk_to_atmostk("../../data/CNFs/equality_model0_and_model4.cnfcc")
atleastk_to_atmostk("../../data/CNFs/equality_model0_and_model5.cnfcc")
atleastk_to_atmostk("../../data/CNFs/equality_model4_and_model0.cnfcc")
atleastk_to_atmostk("../../data/CNFs/equality_model4_and_model4.cnfcc")
atleastk_to_atmostk("../../data/CNFs/equality_model4_and_model5.cnfcc")
atleastk_to_atmostk("../../data/CNFs/equality_model5_and_model0.cnfcc")
atleastk_to_atmostk("../../data/CNFs/equality_model5_and_model4.cnfcc")
atleastk_to_atmostk("../../data/CNFs/equality_model5_and_model5.cnfcc")
