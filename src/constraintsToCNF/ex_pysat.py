# from pysat.card import *
#
# cnf = CardEnc.atleast([1, 2, 3], 2, top_id=1000)
# print(cnf.clauses)


from pysat.card import ITotalizer

# t = ITotalizer(lits=[1, 2, 3], ubound=3, top_id=100)
# print(t.cnf.clauses)
# print(t.rhs[-2])
# print(t.rhs)
# t.delete()
#
#
# path_res_cnf = "../../data/CNFs/test.cnf"
# file_res = open(path_res_cnf, 'w')
# for i in range(10):
#     file_res.write('AAAAAAAAAAAAAAAAAAAAAAA\n')
# file_res.close()
#
# with open(path_res_cnf, 'r+') as fp:
#     lines = fp.readlines()
#     lines.insert(0, 'p cnf\n')
#     fp.seek(0)
#     fp.writelines(lines)

ex = '-927 928 929 930 931 932 933 -934 935 -936 937 938 939 -940 941 942 943 -944 -945 946 947 948 -949 -950 951 952 953 954 955 956 957 958 959 960 -961 962 963 -964 -965 -966 -967 968 969 -970 971 972 -973 -974 975 -976 -977 978 979 -980 981 -982 -983 -984 -985 986 987 988 989 990'
vars = []
for i in ex.split():
    vars.append(int(i))
t = ITotalizer(lits=vars, ubound=74)
print(t.cnf.clauses)
print(t.rhs)
print(len(t.rhs))
