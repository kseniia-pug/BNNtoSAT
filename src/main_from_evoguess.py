from base64 import b85decode
from typings.searchable import Searchable

from algorithm.impl import LogSearch
from algorithm.module.mutation import LogDiv
from algorithm.module.selection import Roulette

from function.impl import GuessAndDetermine
from function.module.solver import MinisatCS
from function.module.budget import AutoBudget
from function.module.measure import SolvingTime

from instance.impl import Instance
from instance.module.encoding import Source
from instance.module.variables import Range

from space.impl import IntervalSet

from util.work_path import WorkPath
from output.impl import OptimizeLogger
from executor.impl import ProcessExecutor

from core.impl import Optimize
from core.module.sampling import Const
from core.module.limitation import WallTime
from core.module.comparator import MinValueMaxSize

if __name__ == '__main__':
    name_cnf = 'equal_models_0_and_0.cnf'
    root_path = WorkPath('examples')
    data_path = root_path.to_path('data')
    cnf_file = data_path.to_file(name_cnf)
    solver_path = root_path.to_path('solvers')

    solver_file = solver_path.to_file('minisat')
    logs_path = root_path.to_path('logs', 'test47')
    solution = Optimize(
        space=IntervalSet(
            indexes=Range(start=1, length=784),
            # by_vector=[0] * (0) + [1] * 784  # Тут должно быть не более 1000 единиц с конца и в сумме == length
            by_vector=Searchable.unpack(b85decode('00001mS`gbS#$6J000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000')) # вместо '000h~Z*0bzV*mgE000' можно вставлять, откуда остановились
        ),
        executor=ProcessExecutor(max_workers=8),
        sampling=Const(size=20, split_into=5),
        instance=Instance(
            encoding=Source(from_file=cnf_file),
        ),
        function=GuessAndDetermine(
            measure=SolvingTime(),
            budget=AutoBudget(),
            solver=MinisatCS(solver_file),
        ),
        algorithm=LogSearch(
            population_size=6,
            mutation=LogDiv(
                max_noise_scale=0.95
            ),
            selection=Roulette(),
            min_update_size=6
        ),
        comparator=MinValueMaxSize(),
        logger=OptimizeLogger(logs_path),
        limitation=WallTime(from_string='15:10:00'),
    ).launch()

    res_path = root_path.to_file('res.txt', 'logs', 'test47')
    file = open(res_path, 'w')
    file.write(name_cnf + '\n')
    for point in solution:
        file.write(point.__str__() + '\n')
    file.close()
