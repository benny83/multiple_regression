require 'csv'
require 'matrix'

# Читаем датасет и записываем в переменную
table = CSV.parse(File.read("vgsales.csv"), headers: true)
simple = table[0..-1]

# Определим вектор оценок коэффициентов регрессии. Согласно методу наименьших квадратов, вектор s получается из выражения: s = (X^(T)*X)^(-1)*X^(T)Y

# Матрица Y - столбец продаж NA
matrix_y = simple.map { |x| x[6].to_f } # SALES_NA

# Матрица X - столбцы зависимостей
# matrix_x = simple.map { |x| [x[2], x[3], x[4]] }.to_a # PLATFORM / YEAR / GENRE
matrix_x = simple.map { |x| [x[3].to_i, x[8].to_f] }.to_a # YEAR / JP_SALES

# Добавляем единичный столбец к матрице X
matrix_x.map { |x| x.unshift(1) } # единичнй столбец
# Транспонируем матрицу
transposed_matrix_x = matrix_x.transpose

# Умножаем матрицы X^(T) и X
matrix_xt_x = Matrix[*transposed_matrix_x] * Matrix[*matrix_x]
# Умножаем матрицы X^(T) и Y
matrix_xt_y = Matrix[*transposed_matrix_x] * Matrix[*matrix_y.map { |x| [x]}]

#Находим обратную матрицу X^(T)*X
inverse_xt_x =  matrix_xt_x.inverse

# Вектор оценок коэффициентов регрессии равен
vector = inverse_xt_x * matrix_xt_y

# Результат
result = "Уравнение регрессии (оценка уравнения регрессии):"
result << "\nY = #{vector.to_a[0][0].to_f} + #{vector.to_a[1][0].to_f} * x1 + #{vector.to_a[2][0].to_f} * x2"
puts result



# Пример
x1 = 2006
x2 = 3.77
example = "\nПример:"
example << "\nYear: #{x1}"
example << "\nJP_SALES: #{x2}"
example << "\nNA_SALES:  #{vector.to_a[0][0].to_f + vector.to_a[1][0].to_f * x1 + vector.to_a[2][0].to_f * x2}"
puts example





# Множественный коэффициент корреляции (Индекс множественной корреляции).
n = table.size

# Выпишем матрицу А
a = table.map { |x| [1, x[6].to_f, x[3].to_i, x[8].to_f] }
matrix_a = Matrix[*a]

transposed_matrix_a = matrix_a.transpose
matrix_at_a = transposed_matrix_a * matrix_a

# Признаки x и y
# 32758598            # 4392.950000000332   # 8687475.520000093
# 1291.0199999999018  # 4392.950000000332   # 2227.3241000000216
# 1291.0199999999018  # 32758598            # 2571807.7400000384

# Дисперсии и среднеквадратические отклонения
Dx_yx1 = a.map { |val| val[2] ** 2 }.sum / n - 32758598 / n
Dy_yx1 = a.map { |val| val[1] ** 2 }.sum / n - 4392.950000000332 / n
s_Dx_yx1 = Math.sqrt(Dx_yx1)
s_Dy_yx1 = Math.sqrt(Dy_yx1)

Dx_yx2 = a.map { |val| val[3] ** 2 }.sum / n - 32758598 / n
Dy_yx2 = Dy_yx1
s_Dx_yx2 = Math.sqrt(Math.sqrt(Dx_yx2**2))
s_Dy_yx2 = s_Dy_yx1

Dx_x1x2 = Dx_yx2
Dy_x1x2 = Dx_yx1
s_Dx_x1x2 = s_Dx_yx2
s_Dy_x1x2 = s_Dx_yx1

# Найдем парные коэффициенты корреляции
r_y_x1 = ((8687475.520000093 / n) - (32758598 * 4392.950000000332) / n) / (s_Dx_yx1 * s_Dy_yx1)
r_y_x2 = ((2227.3241000000216 / n) - (1291.0199999999018 * 4392.950000000332) / n) / (s_Dx_yx2 * s_Dy_yx2)
r_x1_x2 = ((2571807.7400000384 / n) - (1291.0199999999018 * 32758598) / n) / (s_Dx_yx2 * s_Dx_yx1)

# где Δr - определитель матрицы парных коэффициентов корреляции; Δr11 - определитель матрицы межфакторной корреляции.
delta_r = Matrix[*[[1, r_y_x1, r_y_x2], [r_y_x1, 1, r_x1_x2], [r_y_x2, r_x1_x2, 1]]].determinant.abs
delta_r_11 = Matrix[*[[1, r_x1_x2], [r_x1_x2, 1]]].determinant.abs

result = ""
result << "\nКоэффициент множественной корреляции: #{Math.sqrt(1 - delta_r_11/delta_r)}"
result << "\nКоэффициент детерминации: #{Math.sqrt(1 - delta_r_11/delta_r)**2}"
puts result

# Чем ближе этот коэффициент к единице, тем больше уравнение регрессии объясняет поведение Y.
