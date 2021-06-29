import numpy as np
import random as rm
import matplotlib.pyplot as plt

class Ewolution:
    def __init__(self):
        # Values:
        self.n = 10
        self.pop = 20
        self.genMAX = 1000
        self.pC = 0.7
        self.pM = 0.4

    def evaluate(self, x):
        error = 0
        right_slant = np.zeros(2 * self.n)
        left_slant = np.zeros(2 * self.n)

        for i in range(self.n):
            right_slant[x[i] + i] += 1
            left_slant[self.n - x[i] + i] += 1

        for i in range(2 * self.n):  # (N * (N - 1)) / 2
            error += int((right_slant[i] * (right_slant[i] - 1)) / 2)
            error += int((left_slant[i] * (left_slant[i] - 1)) / 2)
        return error

    def individual(self):
        population = np.zeros((self.pop, self.n))
        for i in range(self.pop):
            random = np.random.permutation(self.n)
            for j in range(self.n):
                population[i, j] = random[j]
        return population

    def evolutionary(self):
        population = np.zeros((self.pop, self.n))
        new_population = np.zeros((self.pop, self.n))
        population = self.individual()
        score_E = []
        gen = 0
        best_evaluate = []
        average = []

        for i in range(self.pop):
            tmp = []
            for j in range(self.n):
                tmp.append(int(population[i, j]))
            score_E.append(self.evaluate(tmp))

        best = np.min(score_E)
        best_evaluate.append(best)
        average.append(np.mean(score_E))
        index_best = score_E.index(best)

        while gen < self.genMAX and score_E[index_best] > 0:
            self.selection(population, new_population)
            self.crossover(new_population)
            self.mutation(new_population)
            for i in range(self.pop):
                tmp = []
                for j in range(self.n):
                    tmp.append(int(population[i, j]))
                score_E.append(self.evaluate(tmp))
            best = np.min(score_E)
            best_evaluate.append(best)
            average.append(np.mean(score_E))
            index_best = score_E.index(best)
            population = new_population
            gen += 1

        self.plot(best_evaluate, average)
        print(population)
        return population[best], score_E[index_best], gen

    def plot(self, n_evaluate, s):
        plt.plot(n_evaluate)
        plt.xlabel("generations")
        plt.ylabel("best values evaluate")
        plt.title("Evolutionary Algorithm")
        plt.savefig('best_values_evaluate.png', dpi = 600)
        plt.show()
        plt.figure()
        plt.xlabel("generations")
        plt.ylabel("average")
        plt.plot(s)
        plt.title("Evolutionary Algorithm")
        plt.savefig('average.png', dpi = 600)
        plt.show()

    def selection(self, P, new_population):
        temp_population = []

        self.conversion(temp_population, P)

        i = 0
        while i < self.pop:
            i1 = rm.randint(0, self.pop - 1)
            i2 = rm.randint(0, self.pop - 1)
            if i1 != i2:
                new_individual = self.evaluate(temp_population[i1]) <= self.evaluate(temp_population[i2]) and temp_population[i1] or temp_population[i2]
                new_population[i] = new_individual
                i += 1

    def crossover(self, P_new):
        i = 0
        temp_population = []

        self.conversion(temp_population, P_new)

        while i < self.pop - 2:
            if np.random.random() <= self.pC:
                self.cross(temp_population[i], temp_population[i+1])
            i += 2

        for i in range(self.pop):
            P_new[i] = temp_population[i]

    def cross(self, P1, P2):
        start = np.math.floor(self.n / 3)
        stop = start + 3

        if P1 == P2:
            print("They're the same, I'm leaving out the crossbreeding")
        else:
            map1 = P1[start:stop]
            map2 = P2[start:stop]

            P1[start:stop] = map2
            P2[start:stop] = map1
            duplicates = []

            for i in map1:
                if (i in map2):
                    duplicates.append(i)

            for i in range(len(map1) - 1, -1, -1):
                if (map1[i] in duplicates):
                    map1.remove(map1[i])
                if (map2[i] in duplicates):
                    map2.remove(map2[i])

            j = 0
            k = 0
            counter = 0
            for i in range(len(P1)):
                if counter < start or counter >= stop:
                    if (j < len(map2) and P1[i] in map2):
                        tmp = map2.index(P1[i])
                        P1[i] = map1[tmp]
                        j += 1
                    if (k < len(map2) and P2[i] in map1):
                        tmp = map1.index(P2[i])
                        P2[i] = map2[tmp]
                        k += 1
                counter += 1

    def mutation(self, P_new):
        i = 0
        while i < self.pop:
            if np.random.random() <= self.pM:
                self.mutate(P_new[i])
                i += 1

    def mutate(self, P_new):
        i1 = rm.randint(0, self.n - 1)
        i2 = rm.randint(0, self.n - 1)

        if i1 == i2:
            while i1 == i2:
                i1 = rm.randint(0, self.n - 1)
                i2 = rm.randint(0, self.n - 1)

        number1 = P_new[i1]
        number2 = P_new[i2]

        P_new[i2] = number1
        P_new[i1] = number2

    def conversion(self, list, table):
        for i in range(self.pop):
            tmp = []
            for j in range(self.n):
                tmp.append(int(table[i, j]))
            list.append(tmp)

if __name__== '__main__':

    e = Ewolution()
    [P_best, evaluate_best, gen] = e.evolutionary()
    print("Best population: ", P_best, "best value evaluate: ", evaluate_best, "Number of generations: ", gen)