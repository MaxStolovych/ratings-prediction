import pandas
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


# Reading in the data.

games = pandas.read_csv('./games.csv')

# Making a histogram of all the ratings in the average_rating column.

plt.hist(games['average_rating'])
plt.show()

# Removing any rows without user reviews.

games = games[games['users_rated']>0]

# Removing any rows with missing values.

games = games.dropna(axis=0)

# Initializing the model.

kmeans_model = KMeans(n_clusters=5,random_state=1)

# Getting only the numeric columns from games.

good_columns = games._get_numeric_data()

# Fitting the model using the good columns.

kmeans_model.fit(good_columns)

# Getting the cluster assignments.

labels = kmeans_model.labels_

# Creating a PCA model.

pca_2 = PCA(2)

# Fitting the PCA model on the numeric columns.

plot_columns = pca_2.fit_transform(good_columns)

# Making a scatter plot of each game, shaded according to cluster assignment.

plt.scatter(x=plot_columns[:,0],y=plot_columns[:,1],c=labels)
plt.show()

# I wanna predict average_rating, so now finding out what might be interesting
# for prediction with correlations.

print(games.corr()["average_rating"])

# It shows that average_weight, that indicates complexity of a game, correlates
# best to average_rating. Also I need to remove columns that can only be 
# computed if the average_rating is already known, because they destroy the 
# purpose of classifier, which is to predict the rating without any 
# previous knowledge.

columns = games.columns.tolist()

# Filtering the columns to remove ones not wanted.

columns = [c for c in columns if c not in ['bayes_average_rating', 
                                           'average_rating', 'type', 'name']]

target = 'average_rating'

# Generating the training set, taking 80% of the rows.

train = games.sample(frac=0.8, random_state=1)

# Selecting anything not in the training set and put it in the testing set.

test = games.loc[~games.index.isin(train.index)]

# Initializing the model and fitting it to the training data.

model = LinearRegression()
model.fit(train[columns], train[target])

# Generating predictions for the test set.

predictions = model.predict(test[columns])

# Finding error value between the test predictions and the actual values.

print(mean_squared_error(predictions, test[target]))

# Trying the other way with the random forest algorithm.

model = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, 
                              random_state=1)

# Fitting the model to the data.

model.fit(train[columns], train[target])

# Make predictions

predictions = model.predict(test[columns])

# Compute the error

print(mean_squared_error(predictions, test[target]))