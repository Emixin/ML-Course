class Polynomial:
    def __init__(self, coefficients):
        self.coefficient_list = coefficients

    def evaluate(self, x):
        ans = 0
        l = len(self.coefficient_list)
        for i in range(l):
            coefficient = self.coefficient_list[i]
            value = coefficient * (x ** (l - i - 1))
            ans += value
        return ans

    def add(self, other):
        l1 = len(self.coefficient_list)
        l2 = len(other.coefficient_list)

        new_coefficients = []
        if l1 == l2:
            new_coefficients = [self.coefficient_list[i] + other.coefficient_list[i] for i in range(l1)]

        elif l1 < l2:
            new_coefficient_list_1 = [0 for i in range(l2 - l1)] + self.coefficient_list
            new_coefficients = [new_coefficient_list_1[i] + other.coefficient_list[i] for i in range(l2)]

        else:
            new_coefficient_list_2 = [0 for i in range(l1 - l2)] + other.coefficient_list
            new_coefficients = [self.coefficient_list[i] + new_coefficient_list_2[i] for i in range(l1)]

        return Polynomial(new_coefficients)

    def subtract(self, other):
        l1 = len(self.coefficient_list)
        l2 = len(other.coefficient_list)

        new_coefficients = []
        if l1 == l2:
            new_coefficients = [self.coefficient_list[i] - other.coefficient_list[i] for i in range(l1)]

        elif l1 < l2:
            new_coefficient_list_1 = [0 for i in range(l2 - l1)] + self.coefficient_list
            new_coefficients = [new_coefficient_list_1[i] - other.coefficient_list[i] for i in range(l2)]

        else:
            new_coefficient_list_2 = [0 for i in range(l1 - l2)] + other.coefficient_list
            new_coefficients = [self.coefficient_list[i] - new_coefficient_list_2[i] for i in range(l1)]

        return Polynomial(new_coefficients)



class QuadraticPolynomial(Polynomial):
    def find_roots(self):
        a = self.coefficient_list[0]
        b = self.coefficient_list[1]
        c = self.coefficient_list[2]

        delta = (b ** 2) - (4 * a * c)
        if delta < 0:
            return "no roots"
        else:
            return (-b + (delta ** 0.5)) / (2 * a), (-b - (delta ** 0.5)) / (2 * a)
