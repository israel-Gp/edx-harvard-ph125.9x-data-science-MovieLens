
# Q1 - How many rows and columns are there in the edx dataset? ------------

nrow(edx)

# 9000055

ncol(edx)

# 6

dim(edx)

# 9000055 X 6

# Q2 - How many zeros were given as ratings in the edx dataset? -----------
# How many threes were given as ratings in the edx dataset?

edx %>% 
  count(rating)

# No 0 Ratings
# 2121240 3 Ratings


# Q3 - How many different movies are in the edx dataset? ------------------

edx %>% 
  summarise(across(everything(),n_distinct))

# 10677

# Q4 - How many different users are in the edx dataset? -------------------

# 69878

# Q5 - How many movie ratings are in each of the following genres  --------

edx %>% 
  separate_longer_delim(cols = 'genres',
                        delim = '|') %>% 
  count(genres)

# Drama - 3910127
# Comedy - 3540930
# Thriller - 2325899
# Romance - 1712100

# Q6 - Which movie has the greatest number of ratings? --------------------

edx %>% 
  count(title, sort = TRUE) %>% 
  slice_max(n)

# Pulp Fiction

# Q7 - What are the five most given ratings in order from most to  --------

edx %>% 
  count(rating, sort = TRUE)

# 4, 3, 5, 3.5, 2

# Q8 - True or False: In general, half star ratings are less commo --------

edx %>% 
  as_tibble() %>% 
  mutate(is_half = rating %% 1 == 0.5) %>% 
  count(is_half, sort = TRUE) %>% 
  mutate(prop = n/sum(n))

# TRUE
