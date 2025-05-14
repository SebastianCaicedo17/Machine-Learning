from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import pandas

df = pandas.read_csv("bienetre.csv")
# print(df.head())

####   Normalisation Standard ######
problem = df.drop("target", axis=1)
solution = df["target"]

standard_scaler = StandardScaler()
normalized_problem = standard_scaler.fit_transform(problem)

### K-NN  ###
def get_optimal_k_value(normalized_problem, solution):
    n_folds = 5
    parameters = {"n_neighbors" : [k for k in range(1, 52 ,2)]}
    grid_search_object = GridSearchCV(KNeighborsClassifier(), parameters, cv=n_folds, scoring="f1_macro")
    grid_search_object.fit(normalized_problem, solution)
    print(grid_search_object.best_params_)
    scores = pandas.DataFrame(grid_search_object.cv_results_)
    return scores

get_optimal_k_value(normalized_problem, solution)

