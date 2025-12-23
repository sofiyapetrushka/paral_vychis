from mpi4py import MPI
import numpy as np

def check_square_number(numprocs):
    """Проверяет, является ли число квадратом натурального числа"""
    sqrt_num = int(np.sqrt(numprocs))
    return sqrt_num * sqrt_num == numprocs

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    numprocs = comm.Get_size()
    
    # Проверка, что количество процессов - квадрат натурального числа
    if rank == 0:
        if not check_square_number(numprocs):
            print(f"Ошибка: Количество процессов ({numprocs}) не является квадратом натурального числа")
            print("Используйте число процессов: 1, 4, 9, 16, 25, ...")
            comm.Abort(1)
        else:
            print(f"Запущено на {numprocs} процессах")
    
    comm.barrier()
    
    # Этап 1.1: Создание декартовой топологии
    # Вычисление размеров сетки
    num_row = int(np.sqrt(numprocs))
    num_col = num_row
    
    # Создание декартовой топологии типа "тор"
    comm_cart = comm.Create_cart(
        dims=(num_row, num_col),
        periods=(True, True), 
        reorder=True
    )
    
    # Получение новых рангов и координат в декартовой топологии
    cart_rank = comm_cart.Get_rank()
    cart_coords = comm_cart.Get_coords(cart_rank)
    
    # Вывод информации о процессе
    print(f"Процесс: глобальный ранг = {rank}, декартов ранг = {cart_rank}, "
          f"координаты = ({cart_coords[0]}, {cart_coords[1]})")
    
    comm_cart.barrier()
    if cart_rank == 0:
        print("\n" + "="*60)
        print("Создана декартова топология (тор) размером "
              f"{num_row} x {num_col}")
        print("="*60 + "\n")
    
    # Этап 1.2: Определение соседей процессов
    # Определение соседей по вертикали (направление 0)
    neighbour_up, neighbour_down = comm_cart.Shift(direction=0, disp=1)
    
    # Определение соседей по горизонтали (направление 1)
    neighbour_left, neighbour_right = comm_cart.Shift(direction=1, disp=1)
    
    # Вывод информации о соседях
    print(f"Процесс {cart_rank} ({cart_coords[0]}, {cart_coords[1]}):")
    print(f"  Сосед сверху: {neighbour_up}")
    print(f"  Сосед снизу: {neighbour_down}")
    print(f"  Сосед слева: {neighbour_left}")
    print(f"  Сосед справа: {neighbour_right}")
    
    comm_cart.barrier()
    
    # Этап 1.3: Реализация кольцевого обмена с использованием Sendrecv_replace
    # Используем комбинацию координат для уникального значения
    unique_value = 100 * (cart_coords[0] + 1) + 10 * (cart_coords[1] + 1) + cart_rank
    a = np.array([unique_value], dtype=np.int32)
    
    original_value = a.copy()
    
    if cart_rank == 0:
        print("\n" + "="*60)
        print("Начало кольцевого обмена данных по горизонтали")
        print("="*60)
    
    comm_cart.barrier()
    
    print(f"Процесс {cart_rank}: исходное значение = {a[0]}")
    
    steps = num_col
    
    for step in range(steps):
        # Используем Sendrecv_replace для отправки правому соседу и приема от левого
        comm_cart.Sendrecv_replace(
            [a, MPI.INT], 
            dest=neighbour_right,
            sendtag=step,  
            source=neighbour_left, 
            recvtag=step  
        )
        
        
        if cart_rank == 0:
            print(f"\nШаг {step + 1}/{steps} завершен")
        
        # Вывод промежуточного состояния
        print(f"Процесс {cart_rank} после шага {step + 1}: значение = {a[0]}")
    
    comm_cart.barrier()
    
    # Проверка результата: каждый процесс должен иметь сумму всех оригинальных значений
    # Собираем все оригинальные значения на процессе 0 для проверки
    all_values = comm_cart.gather(original_value[0], root=0)
    
    if cart_rank == 0:
        total_sum = sum(all_values)
        print("\n" + "="*60)
        print("Результаты кольцевого обмена:")
        print(f"Сумма всех оригинальных значений: {total_sum}")
        print("Значения после полного обхода кольца:")
        
        # Соберем финальные значения с всех процессов
        final_values = comm_cart.gather(a[0], root=0)
        
        # Проверим, что все процессы имеют сумму всех значений
        all_correct = True
        for i, val in enumerate(final_values):
            if val != total_sum:
                print(f"Ошибка: процесс {i} имеет значение {val}, ожидается {total_sum}")
                all_correct = False
        
        if all_correct:
            print("Успех! Все процессы содержат сумму всех оригинальных значений")
        print("="*60)
    
    # Дополнительный этап: вертикальный кольцевой обмен
    if cart_rank == 0:
        print("\n\n" + "="*60)
        print("Начало кольцевого обмена данных по вертикали")
        print("="*60)
    
    comm_cart.barrier()
    
    # Сбросим данные к оригинальным значениям
    a[0] = original_value[0]
    
    print(f"Процесс {cart_rank}: сброшено на исходное значение = {a[0]}")
    
    # Кольцевой обмен по вертикали (направление 0)
    steps_vertical = num_row
    
    for step in range(steps_vertical):
        # Используем Sendrecv_replace для отправки нижнему соседу и приема от верхнего
        comm_cart.Sendrecv_replace(
            [a, MPI.INT],  # Буфер для отправки и приема
            dest=neighbour_down,  # Куда отправляем
            sendtag=100 + step,  # Метка отправки (другой диапазон тегов)
            source=neighbour_up,  # Откуда получаем
            recvtag=100 + step  # Метка приема
        )
        
        print(f"Процесс {cart_rank} после вертикального шага {step + 1}: значение = {a[0]}")
    
    comm_cart.barrier()
    
    # Вывод финальных результатов вертикального обмена
    if cart_rank == 0:
        print("\n" + "="*60)
        print("Результаты вертикального кольцевого обмена:")
        
        # Соберем финальные значения с всех процессов
        final_values_vertical = comm_cart.gather(a[0], root=0)
        
        for i, val in enumerate(final_values_vertical):
            print(f"Процесс {i}: финальное значение = {val}")
        print("="*60)
    
    # Пример более сложного обмена: одновременный обмен в обоих направлениях
    if cart_rank == 0:
        print("\n\n" + "="*60)
        print("Демонстрация обмена с соседями во всех направлениях")
        print("="*60)
    
    comm_cart.barrier()
    
    # Подготовим отдельные массивы для каждого направления
    send_up = np.array([cart_rank + 1000], dtype=np.int32)  # Для отправки вверх
    recv_down = np.zeros(1, dtype=np.int32)  # Для приема снизу
    
    send_down = np.array([cart_rank + 2000], dtype=np.int32)  # Для отправки вниз
    recv_up = np.zeros(1, dtype=np.int32)  # Для приема сверху
    
    send_left = np.array([cart_rank + 3000], dtype=np.int32)  # Для отправки влево
    recv_right = np.zeros(1, dtype=np.int32)  # Для приема справа
    
    send_right = np.array([cart_rank + 4000], dtype=np.int32)  # Для отправки вправо
    recv_left = np.zeros(1, dtype=np.int32)  # Для приема слева
    
    # Создаем отдельные запросы для неблокирующих операций
    requests = []
    
    # Неблокирующие отправки и приемы
    req = comm_cart.Isend([send_up, MPI.INT], dest=neighbour_up, tag=0)
    requests.append(req)
    
    req = comm_cart.Irecv([recv_down, MPI.INT], source=neighbour_down, tag=0)
    requests.append(req)
    
    req = comm_cart.Isend([send_down, MPI.INT], dest=neighbour_down, tag=1)
    requests.append(req)
    
    req = comm_cart.Irecv([recv_up, MPI.INT], source=neighbour_up, tag=1)
    requests.append(req)
    
    req = comm_cart.Isend([send_left, MPI.INT], dest=neighbour_left, tag=2)
    requests.append(req)
    
    req = comm_cart.Irecv([recv_right, MPI.INT], source=neighbour_right, tag=2)
    requests.append(req)
    
    req = comm_cart.Isend([send_right, MPI.INT], dest=neighbour_right, tag=3)
    requests.append(req)
    
    req = comm_cart.Irecv([recv_left, MPI.INT], source=neighbour_left, tag=3)
    requests.append(req)
    
    # Ждем завершения всех операций
    MPI.Request.Waitall(requests)
    
    # Вывод результатов обмена
    print(f"Процесс {cart_rank} получил:")
    print(f"  Сверху: {recv_up[0]}")
    print(f"  Снизу: {recv_down[0]}")
    print(f"  Слева: {recv_left[0]}")
    print(f"  Справа: {recv_right[0]}")
    
    comm_cart.barrier()
    
    if cart_rank == 0:
        print("\n" + "="*60)
        print("Программа успешно завершена!")
        print("="*60)

if __name__ == "__main__":
    main()