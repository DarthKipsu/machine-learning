import loadmovielens as reader
import math as m
import numpy as np

# load data
ratings, movie_dictionary, user_ids, item_ids, movie_names = reader.read_movie_lens_data()

# B part #

# calculate Jaccard coefficient for two movie ID's
# (nr. of users who rated both movies / nr. of users who rated at least one of them)
def jaccard_coefficient(movie_id_1, movie_id_2):
    rated_movie1 = ratings[ratings[:, 1] == movie_id_1][:, 0]
    rated_movie2 = ratings[ratings[:, 1] == movie_id_2][:, 0]
    rated_either = len(set(np.concatenate((rated_movie1, rated_movie2), axis=0)))
    rated_both = len(np.intersect1d(rated_movie1, rated_movie2))
    return round(rated_both / rated_either, 3)

# name of a movie from movie id
def name(movie_id):
    return movie_dictionary[movie_id]

# 5 movies with the highest Jaccard coefficient for a movie id
def jaccard_coefficient_top_5(movie_id):
    coefficients = map(lambda x: [x, jaccard_coefficient(movie_id, x)], range(1, 1682))
    sorted_coefficients = sorted(coefficients, key=lambda x: x[1])
    top5 = sorted_coefficients[-6:][:5]
    top5_names = map(lambda x: [name(x[0]), x[1]], top5)
    return list(top5_names)

# Test Jaccard coefficient, this should produce 0.217
print("Jaccard coefficient for ", name(1), " & ", name(2), ": ", jaccard_coefficient(1, 2))

# Print Jaccard coefficient for 'Three Colors: Red' and 'Three Colors: Blue'
print("Jaccard coefficient for ", name(59), " & ", name(60), ": ", jaccard_coefficient(59, 60))

# print the five movies with highest Jaccard coefficient with Taxi Driver
# this opearation is expensive and can take a few minutes, so uncomment only if you need the result.
print("five top Jaccard coefficient movies for Taxi Driver: ", jaccard_coefficient_top_5(23))

# print the five movies with highest Jaccard coefficient with Star Trek: the Wrath of Khan
# this opearation is expensive and can take a few minutes, so uncomment only if you need the result.
print("five top Jaccard coefficient movies for Star Trek: the Wrath of Khan: ", jaccard_coefficient_top_5(228))

# C part #

# calculate similarity measure using the ratings made by users who have rated both movies
def pearson_correlation(movie1, movie2):
    rated_movie1 = [rating[0] for rating in ratings if rating[1] == movie1]
    rated_movie2 = [rating[0] for rating in ratings if rating[1] == movie2]
    rated_both = set(rated_movie1).intersection(rated_movie2)
    if len(rated_both) < 20:
        return 0

    ratings_movie1 = [rating[2] for rating in ratings if rating[1] == movie1 and rating[0] in rated_both]
    ratings_movie2 = [rating[2] for rating in ratings if rating[1] == movie2 and rating[0] in rated_both]
    mean1 = np.mean(ratings_movie1)
    mean2 = np.mean(ratings_movie2)

    #print(sorted(ratings_movie1))
    #print(sorted(ratings_movie2))

    numerator = 0
    denomX = 0
    denomY = 0
    for i in range(len(rated_both)):
        x = ratings_movie1[i] - mean1
        y = ratings_movie2[i] - mean2
        numerator += x * y
        denomX += x * x
        denomY += y * y

    if (denomX == 0 or denomY == 0):
        return 0

    return numerator / m.sqrt(denomX * denomY)

# 5 movies with the highest correlation for a movie id
def pearson_correlation_top5(movie_id):
    coefficients = map(lambda x: [x, pearson_correlation(movie_id, x)], range(1, 1682))
    sorted_coefficients = sorted(coefficients, key=lambda x: x[1])
    top5 = sorted_coefficients[-8:]
    top5_names = map(lambda x: [name(x[0]), x[1]], top5)
    return list(top5_names)

# Print correlation for 'Toy Story' and 'GoldenEye'
print("Correlation for ", name(1), " & ", name(2), ": ", pearson_correlation(1, 2))

# Print correlation for 'Three Colors: Red' and 'Three Colors: Blue'
print("correlation for ", name(59), " & ", name(60), ": ", pearson_correlation(59, 60))

# print the five movies with highest correlation with Taxi Driver
# this opearation is expensive and can take a few minutes, so uncomment only if you need the result.
#print("five top correlation movies for Taxi Driver: ", pearson_correlation_top5(23))

# print the five movies with highest correlation with Star Trek: the Wrath of Khan
# this opearation is expensive and can take a few minutes, so uncomment only if you need the result.
#print("five top correlation movies for Star Trek: the Wrath of Khan: ", pearson_correlation_top5(228))
