from pysat.card import CardEnc, EncType
from pysat.formula import CNFPlus, CNF

# t = ITotalizer(lits=atmost[0], ubound=atmost[1])
        # res_cnf.extend(t.cnf.clauses)
        # res_cnf.append([-t.rhs[-1]])
        # t.delete()

def cnfPlus_to_cnf(path_cnf, path_res_cnf):
    cnf = CNFPlus(from_file=path_cnf)
    res_cnf = CNF(from_clauses=cnf.clauses)
    for atmost in cnf.atmosts:
        res_cnf.extend(CardEnc.atmost(lits=atmost[0], bound=atmost[1], top_id=max(cnf.nv, res_cnf.nv), encoding=EncType.cardnetwrk).clauses)
    res_cnf.to_file(path_res_cnf)
    print('End')


# cnfPlus_to_cnf("../../data/CNFs/equal_models_0_and_0.cnfp", "../../data/CNFs/equal_models_0_and_0_cnf_from_cnf+.cnf")
# cnfPlus_to_cnf("../../data/CNFs/equal_models_0_and_4.cnfp", "../../data/CNFs/equal_models_0_and_4_cnf_from_cnf+.cnf")
# cnfPlus_to_cnf("../../data/CNFs/equal_models_0_and_5.cnfp", "../../data/CNFs/equal_models_0_and_5_cnf_from_cnf+.cnf")
# cnfPlus_to_cnf("../../data/CNFs/equal_models_4_and_0.cnfp", "../../data/CNFs/equal_models_4_and_0_cnf_from_cnf+.cnf")
# cnfPlus_to_cnf("../../data/CNFs/equal_models_4_and_4.cnfp", "../../data/CNFs/equal_models_4_and_4_cnf_from_cnf+.cnf")
# cnfPlus_to_cnf("../../data/CNFs/equal_models_4_and_5.cnfp", "../../data/CNFs/equal_models_4_and_5_cnf_from_cnf+.cnf")
# cnfPlus_to_cnf("../../data/CNFs/equal_models_5_and_0.cnfp", "../../data/CNFs/equal_models_5_and_0_cnf_from_cnf+.cnf")
# cnfPlus_to_cnf("../../data/CNFs/equal_models_5_and_4.cnfp", "../../data/CNFs/equal_models_5_and_4_cnf_from_cnf+.cnf")
# cnfPlus_to_cnf("../../data/CNFs/equal_models_5_and_5.cnfp", "../../data/CNFs/equal_models_5_and_5_cnf_from_cnf+.cnf")

cnfPlus_to_cnf("../../data/CNFs/equal_models_0_and_0_small.cnfp", "../../data/CNFs/equal_models_0_and_0_small_cnf_from_cnf+.cnf")
cnfPlus_to_cnf("../../data/CNFs/equal_models_0_and_4_small.cnfp", "../../data/CNFs/equal_models_0_and_4_small_cnf_from_cnf+.cnf")
cnfPlus_to_cnf("../../data/CNFs/equal_models_0_and_5_small.cnfp", "../../data/CNFs/equal_models_0_and_5_small_cnf_from_cnf+.cnf")
cnfPlus_to_cnf("../../data/CNFs/equal_models_4_and_0_small.cnfp", "../../data/CNFs/equal_models_4_and_0_small_cnf_from_cnf+.cnf")
cnfPlus_to_cnf("../../data/CNFs/equal_models_4_and_4_small.cnfp", "../../data/CNFs/equal_models_4_and_4_small_cnf_from_cnf+.cnf")
cnfPlus_to_cnf("../../data/CNFs/equal_models_4_and_5_small.cnfp", "../../data/CNFs/equal_models_4_and_5_small_cnf_from_cnf+.cnf")
cnfPlus_to_cnf("../../data/CNFs/equal_models_5_and_0_small.cnfp", "../../data/CNFs/equal_models_5_and_0_small_cnf_from_cnf+.cnf")
cnfPlus_to_cnf("../../data/CNFs/equal_models_5_and_4_small.cnfp", "../../data/CNFs/equal_models_5_and_4_small_cnf_from_cnf+.cnf")
cnfPlus_to_cnf("../../data/CNFs/equal_models_5_and_5_small.cnfp", "../../data/CNFs/equal_models_5_and_5_small_cnf_from_cnf+.cnf")


# cnfPlus_to_cnf("../../data/CNFs/check2.cnfp", "../../data/CNFs/check2_cnf_from_cnf+.cnf")
# cnfPlus_to_cnf("../../data/CNFs/check3.cnfp", "../../data/CNFs/check3_cnf_from_cnf+.cnf")
