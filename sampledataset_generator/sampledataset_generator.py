import numpy as np
import pandas as pd

linear = lambda X, coef: np.sum(X * coef, axis=1)

friedman1 = (
    lambda X, coef: 10 * np.sin(np.pi * X[:, 0] * X[:, 1])
    + 20 * (X[:, 2] - 0.5) ** 2
    + 10 * X[:, 3]
    + 5 * X[:, 4]
)

friedman2 = (
    lambda X, coef: (X[:, 0] ** 2 + (X[:, 1] * X[:, 2] - 1 / (X[:, 1] * X[:, 3])) ** 2)
    ** 0.5
)

friedman3 = lambda X, coef: np.arctan(
    (X[:, 1] * X[:, 2] - 1 / (X[:, 1] * X[:, 3])) / X[:, 0]
)


class SampleDatasetGenerator:
    def __init__(
        self,
        n_samples=100,
        n_features=10,
        n_informative=5,
        noise=0.0,
        function=linear,
        independence=1.0,
    ):
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_informative = n_informative
        self.noise = noise
        self.function = function
        self.independence = independence
        self.X = False
        self.Y = False
        self.coef = False

    def generate(self):
        self.X = (
            np.random.randn(self.n_samples, self.n_informative)
            - np.random.rand(self.n_samples, self.n_informative)
            + 0.3
        ) * np.random.randn(self.n_informative)
        X = self.X

        self.coef = np.random.randn(self.n_informative) * 10
        coef = self.coef

        self.Y = self.function(X, coef) + self.noise * np.random.randn(self.n_samples)

        for n in range(self.n_features - self.n_informative):
            x = np.random.randn(self.n_samples) * self.independence
            x += self.X[:, n % self.n_informative] * (1 - self.independence)
            self.X = np.concatenate([self.X, x.reshape(self.n_samples, 1)], axis=1)
            self.coef = np.append(dataset.coef, 0)

    def categoricalize(self, column=-1, labels=[0, 1], classification_ratio=0.5):
        if column >= 0 and column < dataset.X.shape[1]:
            matrix = pd.DataFrame(self.X)
            matrix[column] = categoricalize(
                self.X[:, column],
                labels=labels,
                classification_ratio=classification_ratio,
            )
            self.X = matrix.to_numpy()
        else:
            self.Y = categoricalize(
                self.Y, labels=labels, classification_ratio=classification_ratio
            )

    def fill_randomly(self, value, rate):
        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[1]):
                if np.random.rand() < rate:
                    self.X[np.ix_([i], [j])] = value

    def fill_nan_randomly(self, rate=0.05):
        self.fill_randomly(np.nan, rate=rate)

    def fill_zero_randomly(self, rate=0.9):
        self.fill_randomly(0, rate=rate)


def categoricalize(dat, labels=[0, 1], classification_ratio=0.5):
    tmp_code = 0
    tmp_dat = dat

    res = []
    while tmp_code < (len(labels) - 1) * 2:
        percentile = np.percentile(tmp_dat, q=classification_ratio * 100)
        tmp_res = np.where(dat < percentile, tmp_code, tmp_code + 1)
        res.append(list(tmp_res))
        uniqs, counts = np.unique(tmp_res, return_counts=True)
        mode = uniqs[counts == np.amax(counts)][0]
        tmp_code += 2
        tmp_dat = dat[tmp_res == mode]

    res = ["".join(str(a)) for a in list(np.array(res).T)]
    keys = list(set(res))
    return [labels[keys.index(x)] for x in res]
