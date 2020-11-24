import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sampledataset_generator import generator
import pandas as pd

def experiment(model_name, model):
    for fname, function in [
                            ["linear", generator.linear], 
                            ["freadman1", generator.friedman1]
    ]:
        figure = plt.figure(figsize=(18, 18))
        i = 0
        independences = [0.0, 0.5, 1.0]
        noises = [0, 0.05, 0.1]
        for independence in independences:        
            for noise in noises:
                i += 1
                ax = plt.subplot(len(independences), len(noises), i)
                result_df = {"model":[], "n_features":[], model_name:[]}
                for n_features in [10, 50, 75, 100, 200, 300, 500, 1000]:
                    for _ in range(10):
                        dataset = generator.SampleDatasetGenerator(
                            function=function,
                            independence = independence,
                            noise = noise,
                            n_features = n_features
                        )
                        dataset.generate()
                        X = dataset.X
                        Y = dataset.Y      
                        X_train, X_test, y_train, y_test = train_test_split(X, Y)

                        model.fit(X_train, y_train)
                        if hasattr(model, "best_estimator_"):
                            print(model.best_estimator_)
                        result_df["model"].append(model_name)
                        result_df["n_features"].append(n_features)
                        result_df[model_name].append(model.score(X_test, y_test))

                result_df = pd.DataFrame(result_df)
                for model_name in result_df["model"].unique():
                    df = result_df[result_df["model"] == model_name]
                    df_err = df.groupby("n_features").std()
                    df.groupby("n_features").mean().plot(ax=ax, kind='bar', grid=True, yerr=df_err)
                    ax.set_xlabel('')
                    ax.set_ylabel('R2')
                    ax.set_title("{} independence={} noise={}".format(function, independence, noise))
