from pathlib import Path

from pysat.card import CardEnc, EncType
from pysat.formula import CNFPlus, CNF


# t = ITotalizer(lits=atmost[0], ubound=atmost[1])
# res_cnf.extend(t.cnf.clauses)
# res_cnf.append([-t.rhs[-1]])
# t.delete()

def cnfp_to_cnf(path):
    assert Path(path).suffix == ".cnfp"
    path_res = path[:-len(".cnfp")] + "_from_cnfp.cnf"

    cnf = CNFPlus(from_file=path)
    res_cnf = CNF(from_clauses=cnf.clauses)
    for atmost in cnf.atmosts:
        res_cnf.extend(CardEnc.atmost(lits=atmost[0], bound=atmost[1], top_id=max(cnf.nv, res_cnf.nv),
                                      encoding=EncType.cardnetwrk).clauses)
    res_cnf.to_file(path_res)


cnfp_to_cnf("../../data/CNFs/universal_adversarial_robustness_model0.cnfp")
cnfp_to_cnf("../../data/CNFs/universal_adversarial_robustness_model4.cnfp")
cnfp_to_cnf("../../data/CNFs/universal_adversarial_robustness_model5.cnfp")
cnfp_to_cnf("../../data/CNFs/equality_model0_and_model0.cnfp")
cnfp_to_cnf("../../data/CNFs/equality_model0_and_model4.cnfp")
cnfp_to_cnf("../../data/CNFs/equality_model0_and_model5.cnfp")
cnfp_to_cnf("../../data/CNFs/equality_model4_and_model0.cnfp")
cnfp_to_cnf("../../data/CNFs/equality_model4_and_model4.cnfp")
cnfp_to_cnf("../../data/CNFs/equality_model4_and_model5.cnfp")
cnfp_to_cnf("../../data/CNFs/equality_model5_and_model0.cnfp")
cnfp_to_cnf("../../data/CNFs/equality_model5_and_model4.cnfp")
cnfp_to_cnf("../../data/CNFs/equality_model5_and_model5.cnfp")

cnfp_to_cnf("../../data/CNFs/universal_adversarial_robustness_model0_without_new_vars.cnfp")
cnfp_to_cnf("../../data/CNFs/universal_adversarial_robustness_model4_without_new_vars.cnfp")
cnfp_to_cnf("../../data/CNFs/universal_adversarial_robustness_model5_without_new_vars.cnfp")
cnfp_to_cnf("../../data/CNFs/equality_model0_and_model0_without_new_vars.cnfp")
cnfp_to_cnf("../../data/CNFs/equality_model0_and_model4_without_new_vars.cnfp")
cnfp_to_cnf("../../data/CNFs/equality_model0_and_model5_without_new_vars.cnfp")
cnfp_to_cnf("../../data/CNFs/equality_model4_and_model0_without_new_vars.cnfp")
cnfp_to_cnf("../../data/CNFs/equality_model4_and_model4_without_new_vars.cnfp")
cnfp_to_cnf("../../data/CNFs/equality_model4_and_model5_without_new_vars.cnfp")
cnfp_to_cnf("../../data/CNFs/equality_model5_and_model0_without_new_vars.cnfp")
cnfp_to_cnf("../../data/CNFs/equality_model5_and_model4_without_new_vars.cnfp")
cnfp_to_cnf("../../data/CNFs/equality_model5_and_model5_without_new_vars.cnfp")
