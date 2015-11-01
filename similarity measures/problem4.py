import loadmovielens as reader

ratings, movie_dictionary, user_ids, item_ids, movie_names = reader.read_movie_lens_data()

print("ratings:")
print(ratings)
