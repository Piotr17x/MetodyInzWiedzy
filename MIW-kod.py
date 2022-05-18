from copy import deepcopy
import numpy as np
import random

plik = open('australian.dat')

lines = plik.readlines()

lista = []

for line in lines:
    line_list = line.split()
    lista.append(list(map(lambda x: float(x), line_list)))


# for a in lista[5]:
#     print(a)


def odl(line1, line2):
    wynik = 0
    for a in range(len(line1)):
        wynik += (line1[a] - line2[a]) ** 2

    return wynik ** (1 / 2)


# lista=[[1,2,2.2,0],[1,2,2.3,0], [1,2,32.3,0], [1,2,5,1],[1,2,23,1],[1,2,2.2,1],[1,222,2.2,1],[1,21,32.2,1],[1,23,2.2,1]]
# print(odl(lista[0], lista[1]))
# print(odl(lista[0], lista[2]))
# print(odl(lista[0], lista[3]))

# print('--------------------')
# list1 = []
# list0 = []
# for w in range(1, len(lista)):
#     if lista[w][-1] == 0:
#         list0.append(odl(lista[0], lista[w]))
#     else:
#         list1.append(odl(lista[0], lista[w]))
#
# dict_1 = {0: list0,
#           1: list1,
#           }
# print(dict_1)


print('--------------------')


def wyz_2x2(m):
    return m[0][0] * m[1][1] - m[0][1] * m[1][0]


def smaller_m(macierz, i):
    new_macierz = deepcopy(macierz[1:])
    for row in new_macierz:
        del row[i]
    return new_macierz


def det_kw(macierz):
    num_rows = len(macierz)
    for row in macierz:
        if len(row) != num_rows:
            return "Macierz nie jest kwadratowa"
    if num_rows == 2:
        return wyz_2x2(macierz)
    else:
        wynik = 0
        for i in range(len(macierz)):
            wynik += (-1) ** (2 + i) * macierz[0][i] * det_kw(smaller_m(macierz, i))
        return wynik


matrix = [
    [1, 2, 4],
    [1, 2, 3],
    [2, 2, 3],
]

print(det_kw(matrix))

print('--------------------')


def lot(x, lista):
    l_of_tuple = []
    for row in lista[1:]:
        if row[-1] in x:
            l_of_tuple.append((int(row[-1]), odl(lista[0][:-1], row[:-1])))
    return l_of_tuple


def lot_to_dict(lot):
    new_dict = {}
    for a, b in lot:
        new_dict.setdefault(a, []).append(b)
    return new_dict


dec_list = lot([0, 1], lista)
print(dec_list)

dec_dict = lot_to_dict(dec_list)
print(dec_dict)

print('--------------------')


# knn


def dddd(dec_dict, k):
    dict_wynik = {}
    for wyn in dec_dict.keys():
        dec_dict[wyn].sort()
        dict_wynik.setdefault(wyn, []).append(sum(dec_dict[wyn][:k]))
    return dict_wynik


print(dddd(dec_dict, 1))

print('--------------------')


def met_euk(l1, l2):
    v1 = np.array(l1)
    v2 = np.array(l2)
    return np.linalg.norm(v1 - v2)


def lot_met_euk(x, lista):
    l_of_tuple = []
    for row in lista[1:]:
        if row[-1] in x:
            l_of_tuple.append((int(row[-1]), met_euk(lista[0][:-1], row[:-1])))
    return l_of_tuple


# print(lot_met_euk((0, 1), lista))
print(odl([1, 1], [4, 4]), met_euk([1, 1], [4, 4]))

print('--------------------')

# całkowanie numeryczne
# metoda montecarlo, wiki


def monte_carlo(x1,x2, k, fun):
    inside = 0
    iter = k
    while iter > 0:
        iter -= 1
        x = random.uniform(x1, x2)
        y = random.uniform(0, eval(fun.replace('x','x2')))
        if y <= eval(fun):
            inside += 1
    return (inside/k) * x2*eval(fun.replace('x', 'x2'))


print('monte ', monte_carlo(0,1, 1000, 'x'))


def met_prost(x1, x2, k, fun):
    dx = (x2 - x1) / k
    integr = 0
    for x in range(k):
        x = x * dx + x1
        integr += dx * eval(fun)
    return integr


print('met_prost', met_prost(0, 1, 100, 'x'))


# suma górna suma dolna, metoda prostokątów
# 28 luty 1:10
print('--------------------')


def lista_odl(pnt,lista):
    lista1 = []
    for row in lista:
        lista1.append(odl(pnt[:-1], row[:-1]))
    return lista1


def sr_ciez(lista):
    smallest = sum(lista_odl(lista[0], lista))
    wynik = lista[0]
    for i in range(1, len(lista)):
        if smallest > sum(lista_odl(lista[i], lista)):
            smallest = sum(lista_odl(lista[i], lista))
            wynik = lista[i]
    return wynik

# print(sr_ciez(lista))

print('--------------------KNN')


def asign_rand(lista, zakres):
    for row in lista:
        row.append(random.choice(zakres))
    return lista


def knn(lista_nieozn, zakres):
    asigned_list = asign_rand(lista_nieozn, zakres)
    print(asigned_list)
    changed = True
    srodki_ciez_previous = []
    while(changed):
        srodki_ciez=[]
        for a in zakres:
            srodki_ciez.append(sr_ciez([x for x in asigned_list if x[-1] == a]))
        for row in asigned_list:
            closest_ind = 0
            distance = odl(row[:-1], srodki_ciez[0][:-1])
            for sr in range(len(srodki_ciez)):
                if distance > odl(row[:-1], srodki_ciez[sr][:-1]):
                    closest_ind = sr
                    distance = odl(row[:-1], srodki_ciez[sr][:-1])
            row[-1] = srodki_ciez[closest_ind][-1]
        if(srodki_ciez == srodki_ciez_previous):
            changed = False
        srodki_ciez_previous = srodki_ciez
    return asigned_list


# print(knn([x[:-1] for x in lista], [0, 1]) == lista)


print('--------------------# 2.4.22')

# iloczyn skalarny
def wlasnosci(lista):
    new_list = lista
    sr = sum(new_list)/len(new_list)
    print('Srednia:', sr)
    wariancja = sum([(x-sr)**2 for x in new_list])/len(new_list)
    print('Wariancja:', wariancja)
    print('Odchylenie standardowe:', wariancja**(1/2))


wlasnosci([5, 5, 5, 5])


print('--------------------regresja ')
# y=ax+b  a=b1 b=b0
# xTxB = xTy
# xTx(-1) * xTxB = B

# [2,1],[5,2],[7,3],[8,3] => = b1=5/14 b0=2/7

macierz = [
    [2, 1],
    [5, 2],
    [7, 3],
    [8, 3],
]


def regresja_liniowa(macierz):
    X = np.array([[1, row[0]] for row in macierz])
    y = np.array([[row[1]] for row in macierz])
    beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return beta


print(regresja_liniowa(macierz))

print('-------------------- rozkład qr')
# ortogonalizacja grama
# 1:04 wykład

# proj u (v) = (<u,v> / <u,u>) * u


# u1 = v1
# u2 = v2 - proj u1 (v2)

# e1 = u1 / ||u1||
# e2 = to samo tylko u2

# QR = QQ(transponowane)A = (QQ(transponowane))(transponowane)(transponowane)A = (Q(transponowane)Q)(transponowane)A = I(transponowane)A = A

# jeśli macierz ma wektory liniowo niezależne można sprowadzić ją do trójkątnej
# [[111]
#  [011]
#  [001]]
# niezależne - z jednego nie da sie uzyskać drugiego
# prostopadłe - iloczyn skalarny = 0


def proj(v1, v2):
    v1_np = np.array(v1)
    v2_np = np.array(v2)
    return np.multiply(v1_np, np.sum(v2_np.T.dot(v1))/np.sum(v1_np.T.dot(v1)))


def magnitude(v):
    return np.divide(v, sum([x**2 for x in v])**(1/2))


def rozkład_qr(macierz):
    macierz = np.array(macierz).astype(float)
    u1 = macierz[:, 0]
    U = np.array([u1])
    e1 = magnitude(u1)
    Q = np.array(np.array([e1]).T)
    for v in range(1, len(macierz)):
        u = deepcopy(macierz[:, v])

        for x in range(v):
            u -= proj(U[x], macierz[:, v])
        U = np.append(U, [u], axis=0)
        Q = np.append(Q, np.array([magnitude(u)]).T, axis=1)
    return Q, Q.T.dot(macierz)

matr=[[1, 0],
      [1, 1]]
matr1=[[12, 6, -4],
       [-51, 167, 24],
       [4, -68, -41]]
matr11=[[1, 1, 1, 1, 1],
       [-51, 167, 24],
       [4, -68, -41]]
print(rozkład_qr(matr))

print('------------------------------ wartości własne')

def wart_wlasne(mat):
    A = mat
    for a in range(20):
        Q, R = rozkład_qr(A)
        A = Q.T.dot(A).dot(Q)
    return [A[x, x] for x in range(len(A))]


print(wart_wlasne(matr))
print('------------------------------ wektory własne')

print(np.linalg.eig(matr)[1])

print('------------------------------ SVD')

mat22 =[[5, -1],
        [5, 7]]


def svd_dec(mat):
    macierz = np.asarray(mat)
    w_w = wart_wlasne(macierz.dot(macierz.T))
    w_s=[]
    for n in w_w:
        w_s.append(n**(1/2))
    sigma = np.diag(w_s)
    macierz_w_w = np.linalg.eig(macierz.dot(macierz.T))[1].T*-1
    V = np.asarray(np.array([magnitude(macierz_w_w[0])]).T)
    for i in range(1, len(macierz_w_w)):
        V = np.append(V, np.array([magnitude(macierz_w_w[i])]).T, axis=1)
    U = np.asarray([np.multiply(1/w_s[-1], np.array(macierz.T).dot(V[:, 0]))]).T
    for i in range(1, len(macierz_w_w)):
        r = len(macierz_w_w)-1-i
        U = np.append(U, np.array([np.multiply(1/w_s[r], np.array(macierz.T).dot(V[:, i]))]).T, axis=1)
    return U, sigma, V.T


print(svd_dec(mat22))

print('------------------------------normalizacja bazy')

# pomnożyc z transpozycją

b=[[1, 1, 1, 1, 1, 1, 1, 1],
   [1, 1, 1, 1, -1, -1, -1, -1],
   [1, 1, -1, -1, 0, 0, 0, 0],
   [0, 0, 0, 0, 1, 1, -1, -1],
   [1, -1, 0, 0, 0, 0, 0, 0],
   [0, 0, 1, -1, 0, 0, 0, 0],
   [0, 0, 0, 0, 1, -1, 0, 0],
   [0, 0, 0, 0, 0, 0, 1, -1]]

def normalizacja(mat):
    new_macierz = np.asarray(mat)
    mat_norm = np.asarray([magnitude(mat[0])])
    for wektor in mat[1:]:
        mat_norm = np.append(mat_norm, [magnitude(wektor)], axis=0)
    return mat_norm



print(normalizacja(b))

print('------------------------------ wektor w innej bazie')
wek = np.asarray([[8,6,2,3,4,6,6,5]])

print(wek.dot(normalizacja(b)))

print('------------------------------')
