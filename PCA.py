from datascience import *
import numpy as np


class Principle_Component_Analysis:
    def __init__(self):
        self.data = None
        self.zero_mean_table = None
        self.covarience_matrix = None
        self.eigen_values = None
        self.eigen_vectors = None
        self.transpose_eigen_vector = None
        self.row_zero_mean_data = []
        self.principal_component = None
        self.principal_component_table = None

        return

    def input(self):
        self.data = Table().read_table('dataMain2.csv')

        print('---------- Input Data ----------')
        print(self.data)
        print()

        return

    def create_zero_mean_table(self):
        col_names = self.data.labels
        self.zero_mean_table = Table()

        for label in col_names:
            col_contents = self.data[label] - np.mean(self.data[label])
            self.zero_mean_table.append_column(label, col_contents)

        print('---------- Zero Mean Table ----------')
        print(self.zero_mean_table)
        print()

        return

    def create_covarience_matrix(self):
        self.covarience_matrix = np.cov(self.zero_mean_table.columns)

        print('---------- Covarience Matrix ----------')
        print(self.covarience_matrix)
        print()

        return

    def create_eigen_values_vectors(self):
        self.eigen_values, self.eigen_vectors = np.linalg.eig(self.covarience_matrix)

        print('---------- Eigen Values ----------')
        print(self.eigen_values)
        print('---------- Eigen Vectors ----------')
        print(self.eigen_vectors)
        print()

        return

    def create_transpose_eigen_vector(self):
        self.transpose_eigen_vector = np.zeros(shape=(len(self.eigen_vectors), len(self.eigen_vectors)))

        for i in range(len(self.eigen_vectors)):
            for j in range(len(self.eigen_vectors)):
                self.transpose_eigen_vector[j, i] = self.eigen_vectors[i, j]

        # print(self.transpose_eigen_vector)

        return

    def create_transpose_zero_mean_table(self):
        for label in self.zero_mean_table.labels:
            for no in self.zero_mean_table[label]:
                self.row_zero_mean_data.append(no)

        self.row_zero_mean_data = np.reshape(self.row_zero_mean_data, (len(self.zero_mean_table.labels), int(len(self.row_zero_mean_data) / len(self.zero_mean_table.labels))))

        # print(self.row_zero_mean_data)

        return

    def find_principle_component(self):
        self.principal_component = np.matrix(self.transpose_eigen_vector) * np.matrix(self.row_zero_mean_data)

        self.principal_component_table = Table()

        counter = len(self.principal_component) - 1

        for lable in self.data.labels:
            self.principal_component_table.append_column(lable, self.principal_component.tolist()[counter])
            counter -= 1

        print('---------- Principle Component Analysis ----------')
        print(self.principal_component_table)
        print()

        return


if __name__ == '__main__':
    pca = Principle_Component_Analysis()

    pca.input()
    pca.create_zero_mean_table()
    pca.create_covarience_matrix()
    pca.create_eigen_values_vectors()
    pca.create_transpose_eigen_vector()
    pca.create_transpose_zero_mean_table()
    pca.find_principle_component()
