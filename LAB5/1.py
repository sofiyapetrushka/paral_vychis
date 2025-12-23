from mpi4py import MPI
import numpy as np

def auxiliary_arrays_determination(M, num):
    quotient = M // num
    remainder = M % num
    
    rcounts = np.zeros(num, dtype=int)
    displs = np.zeros(num, dtype=int)
    
    offset = 0
    for i in range(num):
        rcounts[i] = quotient + 1 if i < remainder else quotient
        displs[i] = offset
        offset += rcounts[i]
    
    return rcounts, displs

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    numprocs = comm.Get_size()
    
    # Этап 1.1: Проверка, что numprocs - квадрат натурального числа
    if rank == 0:
        sqrt_numprocs = int(np.sqrt(numprocs))
        if sqrt_numprocs * sqrt_numprocs != numprocs:
            print(f"Ошибка: Количество процессов ({numprocs}) не является квадратом натурального числа")
            comm.Abort(1)
    
    # Вычисление размеров сетки
    num_col = int(np.sqrt(numprocs))
    num_row = num_col
    
    # Этап 1.2: Создание коммуникаторов для строк и столбцов
    color_col = rank % num_col
    comm_col = comm.Split(color_col, rank)
    
    color_row = rank // num_col
    comm_row = comm.Split(color_row, rank)
    
    rank_row = comm_row.Get_rank()
    rank_col = comm_col.Get_rank()
    
    # Этап 1.3: Чтение параметров и распределение размеров блоков
    if rank == 0:
        # Чтение размеров матрицы из файла
        with open('in.dat', 'r') as f:
            lines = f.readlines()
            M = int(lines[0].strip())  # строки матрицы
            N = int(lines[1].strip())  # столбцы матрицы
    else:
        M = N = 0
    
    # Рассылка размеров матрицы всем процессам
    M = comm.bcast(M, root=0)
    N = comm.bcast(N, root=0)
    
    # Определение массивов для распределения
    if rank == 0:
        rcounts_M, displs_M = auxiliary_arrays_determination(M, num_row)
        rcounts_N, displs_N = auxiliary_arrays_determination(N, num_col)
    else:
        rcounts_M, displs_M = None, None
        rcounts_N, displs_N = None, None
    
    # Рассылка массивов rcounts_N и displs_N
    rcounts_N = comm.bcast(rcounts_N, root=0) if rank == 0 else None
    displs_N = comm.bcast(displs_N, root=0) if rank == 0 else None
    
    rcounts_N = comm.bcast(rcounts_N, root=0)
    displs_N = comm.bcast(displs_N, root=0)
    
    # Определение размера блока для каждого процесса
    if rank_col == 0:  # первый столбец
        N_part = np.empty(1, dtype=int)
        if rank_row == 0:
            N_part_scatter = np.array([rcounts_N[i] for i in range(num_col)])
        else:
            N_part_scatter = None
        
        comm_row.Scatter([N_part_scatter, MPI.INT], [N_part, MPI.INT], root=0)
    else:
        N_part = np.empty(1, dtype=int)
    
    # Broadcast размера блока внутри столбца
    comm_col.Bcast([N_part, MPI.INT], root=0)
    N_part = int(N_part[0])
    
    # Аналогично для M_part
    if rank_row == 0:  # первая строка
        M_part = np.empty(1, dtype=int)
        if rank_col == 0:  # процесс (0,0) - root для всей операции
            M_part_scatter = np.array([rcounts_M[i] for i in range(num_row)])
        else:
            M_part_scatter = None
        
        comm_col.Scatter([M_part_scatter, MPI.INT], [M_part, MPI.INT], root=0)
    else:
        M_part = np.empty(1, dtype=int)
    
    comm_row.Bcast([M_part, MPI.INT], root=0)
    M_part = int(M_part[0])
    
    # Этап 1.4: Распределение данных матрицы A и вектора X
    
    # Часть 1: Распределение матрицы A
    if rank == 0:
        # Чтение всей матрицы A
        A = np.loadtxt('AData.dat', dtype=np.float64)
        if A.ndim == 1:
            A = A.reshape(M, N)
    else:
        A = None
    
    # Создание временных коммуникаторов для распределения матрицы A
    group_world = comm.Get_group()
    
    # Распределение A по блокам - сложная процедура
    A_part = np.zeros((M_part, N_part), dtype=np.float64)
    
    if rank == 0:
        # Для каждого блока матрицы создаем временную группу
        for i in range(num_row):
            for j in range(num_col):
                # Определяем процесс-получатель блока (i,j)
                dest_rank = i * num_col + j
                
                if dest_rank == 0:
                    # Процесс 0 просто копирует свой блок
                    start_row = displs_M[i]
                    start_col = displs_N[j]
                    A_part[0:M_part, 0:N_part] = A[start_row:start_row+M_part, start_col:start_col+N_part]
                else:
                    # Создаем группу из двух процессов: 0 и dest_rank
                    ranks = [0, dest_rank]
                    group_temp = group_world.Range_incl([(ranks[0], ranks[1], 1)])
                    comm_temp = comm.Create(group_temp)
                    
                    if comm_temp != MPI.COMM_NULL:
                        if rank == 0:
                            # Подготовка данных для отправки
                            start_row = displs_M[i]
                            start_col = displs_N[j]
                            block_data = A[start_row:start_row+rcounts_M[i], start_col:start_col+rcounts_N[j]]
                            
                            # Отправка данных
                            comm_temp.Send([block_data, MPI.DOUBLE], dest=1, tag=0)
                        
                        comm_temp.Free()
                    
                    group_temp.Free()
    else:
        # Получение блока матрицы
        # Определяем, в каком блоке мы находимся
        i = rank_row
        j = rank_col
        
        # Создаем группу из двух процессов: 0 и текущего
        ranks = [0, rank]
        group_temp = group_world.Range_incl([(ranks[0], ranks[1], 1)])
        comm_temp = comm.Create(group_temp)
        
        if comm_temp != MPI.COMM_NULL:
            # Получение данных
            comm_temp.Recv([A_part, MPI.DOUBLE], source=0, tag=0)
            comm_temp.Free()
        
        group_temp.Free()
    
    # Часть 2: Распределение вектора X
    if rank == 0:
        # Чтение вектора X
        x = np.loadtxt('xData.dat', dtype=np.float64)
    else:
        x = None
    
    # Распределение x по первой строке процессов
    x_part = np.zeros(N_part, dtype=np.float64)
    
    if rank_row == 0:  # процессы в первой строке
        if rank_col == 0:  # процесс (0,0) - root
            # Подготовка данных для scatter
            sendbuf_x = []
            counts_x = []
            displs_x = []
            
            offset = 0
            for j in range(num_col):
                count = rcounts_N[j]
                sendbuf_x.append(x[offset:offset+count])
                counts_x.append(count)
                displs_x.append(offset)
                offset += count
            
            sendbuf_x = np.concatenate(sendbuf_x)
            counts_x = np.array(counts_x, dtype=int)
            displs_x = np.array(displs_x, dtype=int)
        else:
            sendbuf_x = None
            counts_x = None
            displs_x = None
        
        # Scatterv внутри строки
        recvbuf_x = np.zeros(N_part, dtype=np.float64)
        comm_row.Scatterv([sendbuf_x, counts_x, displs_x, MPI.DOUBLE], 
                         [recvbuf_x, MPI.DOUBLE], root=0)
        x_part = recvbuf_x
    
    # Broadcast вектора x внутри каждого столбца
    comm_col.Bcast([x_part, MPI.DOUBLE], root=0)
    
    # Этап 1.5: Локальные вычисления и сбор результата
    
    # 1. Локальные вычисления
    b_part_temp = np.dot(A_part, x_part)
    
    # 2. Суммирование вдоль строк
    b_part = np.zeros(M_part, dtype=np.float64)
    
    # Корневой процесс в каждой строке (первый столбец)
    root_in_row = 0
    comm_row.Reduce([b_part_temp, MPI.DOUBLE], [b_part, MPI.DOUBLE], 
                    op=MPI.SUM, root=root_in_row)
    
    # 3. Сбор итогового вектора на процессе 0
    if rank_col == 0:  # процессы в первом столбце
        # Подготовка для Gatherv
        if rank_row == 0:  # процесс 0 - root для gather
            recvcounts = rcounts_M
            recvdispls = displs_M
            b_full = np.zeros(M, dtype=np.float64)
        else:
            recvcounts = None
            recvdispls = None
            b_full = None
        
        # Gatherv внутри столбца
        comm_col.Gatherv([b_part, MPI.DOUBLE], 
                        [b_full, recvcounts, recvdispls, MPI.DOUBLE], 
                        root=0)
        
        # Вывод результата на процессе 0
        if rank == 0:
            print("Результат умножения матрицы на вектор:")
            print(b_full)
            
            # Сохранение результата в файл
            np.savetxt('result.dat', b_full, fmt='%.6f')
    else:
        b_full = None

if __name__ == "__main__":
    main()