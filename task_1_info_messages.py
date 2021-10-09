import math
import pprint
import matplotlib.pyplot as plt

with open('task_1_info_messages.txt') as inf:
    s = inf.readline().strip().split('\t')

    n_letters = int(s[0].split(': ')[1])  # Число букв
    bit_length = int(s[1].split(': ')[1].split(' ')[0])  # Разрядность кода
    q = float(s[2].split(': ')[1])  # q = P_error_01 = P_error_10
    p_erase = float(s[4].split(': ')[1])  # P_erase
    m = int(s[5].split(': ')[1])  # Число посылок

    all_packages = dict()  # n = 216 букв, для каждой m = 17 посылок
    for n in range(n_letters):
        all_packages[n+1] = dict()

    for i in range(m):
        line = inf.readline().strip().split(': ')[1].split(' ')
        for n in range(n_letters):
            all_packages[n+1][i+1] = line[n]

# pprint.pprint(all_packages)

alphabet = dict()  # Используемый алфавит
with open('table1.txt', encoding="utf-8") as table:
    for line in table.readlines():
        symbol = line.rstrip().split(': ')[0]
        code = line.rstrip().split(': ')[1]
        alphabet[symbol] = code
# pprint.pprint(alphabet)

# Вычислим априорное распределение вероятностей исходных букв алфавита:
# 1) Все символы равновероятны
p_x_i_1 = dict()
for symbol in alphabet:
    p_x_i_1[symbol] = 1 / len(alphabet)
print('Априорное распределение вероятностей исходных букв алфавита.\n'
      'Все символы равновероятны.')
pprint.pprint(p_x_i_1)

# 2) Вероятности букв задаются исходя из известной информации о частоте букв в русском алфавите
symbols_frequency = dict()
sum_frequency = 0.0
with open('table2.txt', encoding="utf-8") as inf:
    for line in inf.readlines():
        symbol = line.rstrip().split(': ')[0]
        frequency = float(line.rstrip().split(': ')[1])
        symbols_frequency[symbol] = frequency
        sum_frequency += frequency
# print(sum_frequency)

p_x_i_2 = dict()
count = 0
for symbol in alphabet:
    if symbol in symbols_frequency:
        p_x_i_2[symbol] = symbols_frequency[symbol] / 100.0
    else:
        p_x_i_2[symbol] = (100.0 - sum_frequency) / 100.0
        count += 1
for symbol in alphabet:
    if not(symbol in symbols_frequency):
        p_x_i_2[symbol] /= count
print('\nАприорное распределение вероятностей исходных букв алфавита.\n'
      'Вероятности букв задаются исходя из известной информации о частоте букв в русском алфавите.')
pprint.pprint(p_x_i_2)

# Вычислить апостериорное распределение вероятностей
all_p_x_i = dict()
all_p_x_i_y_j = dict()
for n in range(n_letters):
    all_p_x_i[n+1] = p_x_i_1
    # all_p_x_i[n+1] = p_x_i_2
    all_p_x_i_y_j[n+1] = dict()
    for (i, code) in all_packages[n+1].items():
        p_y_j_x_i = dict()
        for (symbol, symbol_code) in alphabet.items():
            n_erase = 0  # количество стёртых разрядов
            t = 0  # количество разрядов, в которых произошла ошибка
            for k in range(len(code)):
                if code[k] == '-':
                    n_erase += 1
                elif code[k] != symbol_code[k]:
                    t += 1
            p_y_j_x_i[symbol] = pow(1-q, bit_length-n_erase-t) * pow(q, t) * pow(1-p_erase, n_erase)

        p_y_j = 0.0
        for symbol in alphabet:
            p_y_j += p_y_j_x_i[symbol] * all_p_x_i[n+1][symbol]

        all_p_x_i_y_j[n+1][i] = dict()
        for symbol in alphabet:
            all_p_x_i_y_j[n+1][i][symbol] = p_y_j_x_i[symbol] * all_p_x_i[n+1][symbol] / p_y_j

        all_p_x_i[n+1] = all_p_x_i_y_j[n+1][i]

# после 1-й, 2-й и m-й передач для каждой s-ой буквы сообщения
s = 8
print('\nАпостериорное распределение вероятностей после 1-й передачи '
      'для s-ой (s = ' + str(s) + ') буквы сообщения)')
pprint.pprint(all_p_x_i_y_j[s][1])
print('\nАпостериорное распределение вероятностей после 2-й передачи '
      'для s-ой (s = ' + str(s) + ') буквы сообщения)')
pprint.pprint(all_p_x_i_y_j[s][2])
print('\nАпостериорное распределение вероятностей после ' + str(m) + '-й передачи '
      'для s-ой (s = ' + str(s) + ') буквы сообщения)')
pprint.pprint(all_p_x_i_y_j[s][m])

# Построить график изменения апостериорного распределения вероятностей
# на примере любой l-ой передаваемой буквы сообщения
l = 8
for (pack_ind, p_x_i_y_j) in all_p_x_i_y_j[l].items():
    plt.figure(figsize=(7, 4))
    plt.axes([0.08, 0.12, 0.90, 0.85])
    plt.plot([i+1 for i in range(len(p_x_i_y_j))], p_x_i_y_j.values())
    plt.xlim([0, 90])
    plt.ylim([-0.02, 1.02])
    plt.xticks(range(5, 90, 5))
    # plt.title('Апостериорное распределение вероятностей для l-ой (l = ' + str(l) +
    #           ')\n передаваемой буквы сообщения после ' + str(pack_ind) + '-й передачи')
    # (n передач => n графиков друг под другом,
    # на графике по оси X – номер символа, по оси Y – вероятность)
    plt.xlabel('Номер символа')
    plt.ylabel('Вероятность')
    plt.savefig('part_1_1_' + str(pack_ind) + '.png')
    plt.close()

# По максимуму апостериорной вероятности определить наиболее вероятные буквы
messages = dict()
for i in range(m):
    messages[i+1] = ''
for (n, p_x_i_y_j) in all_p_x_i_y_j.items():
    for (i, prob_list) in p_x_i_y_j.items():
        symbol_max_prob = ''
        max_prob = 0.0
        for (symbol, prob) in prob_list.items():
            if prob > max_prob:
                max_prob = prob
                symbol_max_prob = symbol
        messages[i] += symbol_max_prob

# Составить вариант исходного переданного сообщения для 1-й, 2-й и m-й посылок
print(messages[1])
print(messages[2])
print(messages[m])

# Выбрать в посылаемом сообщении произвольную букву (под номером s),
# далее все вычисления будут относиться к этой букве
s = 8

# Определить апостериорные вероятности,
# рассматривая каждую передачу независимо от другой
p_x_i = p_x_i_1
# p_x_i = p_x_i_2
all_p_x_i_y_j = dict()
all_p_y_j = dict()
for (i, code) in all_packages[s].items():
    p_y_j_x_i = dict()
    for (symbol, symbol_code) in alphabet.items():
        n_erase = 0  # количество стёртых разрядов
        t = 0  # количество разрядов, в которых произошла ошибка
        for k in range(len(code)):
            if code[k] == '-':
                n_erase += 1
            elif code[k] != symbol_code[k]:
                t += 1
        p_y_j_x_i[symbol] = pow(1 - q, bit_length - n_erase - t) * pow(q, t) * pow(1 - p_erase, n_erase)

    p_y_j = 0.0
    for symbol in alphabet:
        p_y_j += p_y_j_x_i[symbol] * p_x_i[symbol]
    all_p_y_j[i] = p_y_j

    all_p_x_i_y_j[i] = dict()
    for symbol in alphabet:
        all_p_x_i_y_j[i][symbol] = p_y_j_x_i[symbol] * p_x_i[symbol] / p_y_j

# Определить условные энтропии H_X_y_j на сообщения y_j для s-ой буквы
all_H_X_y_j = dict()
for i in range(m):
    all_H_X_y_j[i + 1] = 0.0
    for prob in all_p_x_i_y_j[i + 1].values():
        all_H_X_y_j[i + 1] += prob * math.log2(prob)
    all_H_X_y_j[i + 1] *= -1.0
print('\nУсловные энтропии для s-ой (s = ' + str(s) + ') буквы сообщения:')
pprint.pprint(all_H_X_y_j)

# Определить среднее количество информации I_X_y_j об X, содержащееся y_j для s-ой буквы
all_I_X_y_j = dict()
for i in range(m):
    all_I_X_y_j[i + 1] = 0.0
    for (symbol, prob) in all_p_x_i_y_j[i + 1].items():
        all_I_X_y_j[i + 1] += prob * math.log2(p_x_i[symbol])
    all_I_X_y_j[i + 1] *= -1.0
    all_I_X_y_j[i + 1] -= all_H_X_y_j[i + 1]
print('\nСреднее количество информации для s-ой (s = ' + str(s) + ') буквы сообщения:')
pprint.pprint(all_I_X_y_j)

# Определить среднюю условную энтропию H_X_Y для s-ой буквы
H_X_Y = 0.0
for i in range(m):
    H_X_Y += all_p_y_j[i + 1] * all_H_X_y_j[i + 1]
print('\nСредняя условная энтропия для s-ой (s = ' + str(s) + ') буквы сообщения = ' + str(H_X_Y))

# Определить среднюю взаимную информацию I_X_Y для s-ой буквы
I_X_Y = 0.0
for i in range(m):
    I_X_Y += all_p_y_j[i + 1] * all_I_X_y_j[i + 1]
print('\nСредняя взаимная информация для s-ой (s = ' + str(s) + ') буквы сообщения = ' + str(I_X_Y))

# Построить график изменения условной энтропии H(X/y_j) от номера посылки
plt.figure(figsize=(7, 4))
plt.axes([0.08, 0.12, 0.90, 0.85])
plt.plot([i for i in all_H_X_y_j.keys()], all_H_X_y_j.values())
plt.xlim([0, 18])
plt.ylim([0.0, 5.0])
plt.xticks(range(1, 18, 1))
# plt.title('График изменения условной энтропии H(X/y_j) для s-ой (s = ' + str(s) +
#           ') передаваемой буквы сообщения')
plt.xlabel('Номер посылки')
plt.ylabel('Условная энтропия (бит)')
plt.savefig('part_1_2_1.png')
plt.close()

# Построить график изменения количества информации I(X:y_j) от номера посылки
plt.figure(figsize=(7, 4))
plt.axes([0.08, 0.12, 0.90, 0.85])
plt.plot([i for i in all_I_X_y_j.keys()], all_I_X_y_j.values())
plt.xlim([0, 18])
plt.ylim([0.0, 5.0])
plt.xticks(range(1, 18, 1))
# plt.title('График изменения количества информации I(X:y_j) для s-ой (s = ' + str(s) +
#           ') передаваемой буквы сообщения')
plt.xlabel('Номер посылки')
plt.ylabel('Количество информации (бит)')
plt.savefig('part_1_2_2.png')
plt.close()

# Рассмотрим m (m = 17) передач сообщений как передачу одного большого сообщения,
# в котором каждый символ многократно (m-кратно) дублируется.
# При этом новый алфавит по сути – m-кратное дублирование старого алфавита.

# Вычислить апостериорное распределение вероятностей для каждой l-ой буквы сообщения
l = 8
all_p_x_new_i_y_new = dict()
for n in range(n_letters):
    p_x_new_i = p_x_i_1
    # p_x_new_i = p_x_i_2
    p_y_new_x_new_i = dict()
    for (symbol, symbol_code) in alphabet.items():
        p_y_new_x_new_i[symbol] = 1.0
        for code in all_packages[n+1].values():
            n_erase = 0  # количество стёртых разрядов
            t = 0  # количество разрядов, в которых произошла ошибка
            for k in range(len(code)):
                if code[k] == '-':
                    n_erase += 1
                elif code[k] != symbol_code[k]:
                    t += 1
            p_y_new_x_new_i[symbol] *= \
                pow(1 - q, bit_length - n_erase - t) * pow(q, t) * pow(1 - p_erase, n_erase)

    p_y_new = 0.0
    for symbol in alphabet:
        p_y_new += p_y_new_x_new_i[symbol] * p_x_new_i[symbol]

    all_p_x_new_i_y_new[n+1] = dict()
    for symbol in alphabet:
        all_p_x_new_i_y_new[n+1][symbol] = p_y_new_x_new_i[symbol] * p_x_new_i[symbol] / p_y_new

print('\nАпостериорное распределение вероятностей для l-ой (l = ' + str(l) + ') буквы сообщения)')
pprint.pprint(all_p_x_new_i_y_new[l])

# Построить график апостериорного распределения вероятностей
# на примере l-ой передаваемой буквы сообщения
plt.figure(figsize=(7, 4))
plt.axes([0.08, 0.12, 0.90, 0.85])
plt.plot([i + 1 for i in range(len(all_p_x_new_i_y_new[l]))], all_p_x_new_i_y_new[l].values())
plt.xlim([0, 90])
plt.ylim([-0.02, 1.02])
plt.xticks(range(5, 90, 5))
# plt.title('Апостериорное распределение вероятностей для l-ой (l = ' + str(l) +
#           ')\n передаваемой буквы сообщения')
# на графике по оси X – номер символа, по оси Y – вероятность)
plt.xlabel('Номер символа')
plt.ylabel('Вероятность')
plt.savefig('part_2_1.png')
plt.close()

# По максимуму апостериорной вероятности определить наиболее вероятные буквы
message = ''
for p_x_new_i_y_new in all_p_x_new_i_y_new.values():
    symbol_max_prob = ''
    max_prob = 0.0
    for (symbol, prob) in p_x_new_i_y_new.items():
        if prob > max_prob:
            max_prob = prob
            symbol_max_prob = symbol
    message += symbol_max_prob

# Составить вариант исходного переданного сообщения
print(message)

# Выбрать в посылаемом сообщении ту же букву, что и использовалась в п. 1.2,
# далее все вычисления будут относиться к этой букве
s = 8

# Определить апостериорные вероятности для s-ой буквы
p_x_new_i_y_new = dict()
p_x_new_i = p_x_i_1
# p_x_new_i = p_x_i_2
p_y_new_x_new_i = dict()
for (symbol, symbol_code) in alphabet.items():
    p_y_new_x_new_i[symbol] = 1.0
    for code in all_packages[s].values():
        n_erase = 0  # количество стёртых разрядов
        t = 0  # количество разрядов, в которых произошла ошибка
        for k in range(len(code)):
            if code[k] == '-':
                n_erase += 1
            elif code[k] != symbol_code[k]:
                t += 1
        p_y_new_x_new_i[symbol] *= \
            pow(1 - q, bit_length - n_erase - t) * pow(q, t) * pow(1 - p_erase, n_erase)

p_y_new = 0.0
for symbol in alphabet:
    p_y_new += p_y_new_x_new_i[symbol] * p_x_new_i[symbol]

p_x_new_i_y_new = dict()
for symbol in alphabet:
    p_x_new_i_y_new[symbol] = p_y_new_x_new_i[symbol] * p_x_new_i[symbol] / p_y_new

# Определить условную энтропию H_X_new_y_new на сообщение y_new для s-ой буквы
H_X_new_y_new = 0.0
for prob in p_x_new_i_y_new.values():
    H_X_new_y_new += prob * math.log2(prob)
H_X_new_y_new *= -1.0
print('\nУсловная энтропия для s-ой (s = ' + str(s) + ') буквы сообщения = ' + str(H_X_new_y_new))

# Определить среднее количество информации I_X_new_y_new об X_new, содержащееся y_new для s-ой буквы
I_X_new_y_new = 0.0
for (symbol, prob) in p_x_new_i_y_new.items():
    I_X_new_y_new += prob * math.log2(p_x_new_i[symbol])
I_X_new_y_new *= -1.0
I_X_new_y_new -= H_X_new_y_new
print('\nСреднее количество информации для s-ой (s = ' + str(s) +
      ') буквы сообщения = ' + str(I_X_new_y_new))

# Определить среднюю условную энтропию H_X_new_Y_new для s-ой буквы
H_X_new_Y_new = p_y_new * H_X_new_y_new
print('\nСредняя условная энтропия для s-ой (s = ' + str(s) + ') буквы сообщения = ' + str(H_X_new_Y_new))

# Определить среднюю взаимную информацию I_X_new_Y_new для s-ой буквы
I_X_new_Y_new = p_y_new * I_X_new_y_new
print('\nСредняя взаимная информация для s-ой (s = ' + str(s) +
      ') буквы сообщения = ' + str(I_X_new_Y_new))
