import loadmovielens as reader

# load data
ratings, movie_dictionary, user_ids, item_ids, movie_names = reader.read_movie_lens_data()

# B part #

# calculate Jaccard coefficient for two movie ID's
# (nr. of users who rated both movies / nr. of users who rated at least one of them)
def jaccard_coefficient(movie1, movie2):
    rated_movie1 = [movie[0] for movie in ratings if movie[1] == movie1]
    rated_movie2 = [movie[0] for movie in ratings if movie[1] == movie2]
    rated_both = len(set(rated_movie1).intersection(rated_movie2))
    rated_either = len(set(rated_movie1 + rated_movie2))
    return rated_both / rated_either

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

# this opearation is expensive and can take a few minutes, so uncomment only if you need the result.
# print the five movies with highest Jaccard coefficient with Taxi Driver
# print(jaccard_coefficient_top_5(23))

# this opearation is expensive and can take a few minutes, so uncomment only if you need the result.
# print the five movies with highest Jaccard coefficient with Star Trek: the Wrath of Khan
# print(jaccard_coefficient_top_5(228))
