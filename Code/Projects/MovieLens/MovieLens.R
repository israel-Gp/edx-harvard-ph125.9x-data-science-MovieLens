# Install Libraries -------------------------------------------------------
tictoc::tic()

if (!require("pacman"))
  install.packages("pacman") # Package Management
pacman::p_load(
  tidyverse,  # Standard Package
  caret, # Machine Learning Package
  klaR,  # Factor Clustering
  lubridate,  # Date-time manipulation
  janitor,  # Data Cleaning
  ggrepel,  # GGplot manipulation
  ggcorrplot,  # GGplot correlation visuals
  ggridges , # GGplot manipulation
  ggh4x , # GGplot manipulation
  latex2exp, # GGplot with Latex
  Boruta,  # Feature Selection wrapper
  patchwork,  # GGplot manipulation
  varrank, # Feature Selection wrapper
  reshape, # Data manipulation
  recommenderlab, # Matrix Factorization
  cluster, # Kmeans Clustering
  parallel, # Parallel Processing
  doParallel, # Parallel Processing
  furrr, # Parallel Mapping
  factoextra, # Clustering Visuals
  rcompanion, # Categorical Correlations
  update = TRUE # Attempt package update if previously installed
)

# Read Data ---------------------------------------------------------------

##########################################################
# Create edx and final_holdout_test sets
##########################################################

# Note: this process could take a couple of minutes

if (!require(tidyverse))
  install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if (!require(caret))
  install.packages("caret", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if (!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip",
                dl)

ratings_file <- "ml-10M100K/ratings.dat"
if (!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if (!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(
    userId = as.integer(userId),
    movieId = as.integer(movieId),
    rating = as.numeric(rating),
    timestamp = as.integer(timestamp)
  )

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind = "Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(
  y = movielens$rating,
  times = 1,
  p = 0.1,
  list = FALSE
)
edx <- movielens[-test_index, ]
temp <- movielens[test_index, ]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>%
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Restart Libraries -------------------------------------------------------

pacman::p_unload(pacman::p_loaded(), character.only = TRUE)

library(klaR) # Factor Clustering
library(lubridate) # Date-time manipulation
library(janitor) # Data Cleaning
library(ggrepel) # GGplot manipulation
library(ggcorrplot) # GGplot correlation visuals
library(ggridges) # GGplot manipulation
library(ggh4x)  # GGplot manipulation
library(latex2exp) # GGplot with Latex
library(Boruta) # Feature Selection wrapper
library(patchwork) # GGplot manipulation
library(varrank) # Feature Selection wrapper
library(reshape) # Data manipulation
library(recommenderlab) # Matrix Factorization
library(cluster) # Kmeans Clustering
library(factoextra) # Clustering Visuals
library(scales) # GGplot Manipulation
library(broom) # Data Tidying
library(caret) # Machine Learning Package
library(parallel) # Parallel Processing
library(doParallel) # Parallel Processing
library(furrr) # Parallel Mapping
library(rcompanion) # Categorical Correlations 
library(tidyverse) # Standard Package

# Cleanup Functions -------------------------------------------------------

clear_memory <- function(keep = NULL) {
  # Clear Global Environment
  # Keep Functions, exceptions and excluded items
  if (is.null(keep))
    remove <-
      setdiff(ls(envir = .GlobalEnv), lsf.str(envir = .GlobalEnv))
  else
    remove <-
      str_subset(negate = TRUE, setdiff(ls(envir = .GlobalEnv),
                                        lsf.str(envir = .GlobalEnv)),
                 paste0('^(', paste(keep, collapse = '|'), ')$'))
  rm(list = remove,
     envir = .GlobalEnv)
  
  # Clear RStudio Plots
  tryCatch({
    dev.off(dev.list()["RStudioGD"])
    message('All plots cleared')
  },
  error = function(e) {
    message('No plots to clear')
  })
  
  # Clear Memory
  invisible(gc(reset = TRUE, full = TRUE))
  message('Memory Cleared')
}

# Pre-split Data Cleaning -------------------------------------------------

edx_clean <- edx %>%
  as_tibble() %>% # Readability in Console
  clean_names() %>%  # Best Names
  mutate(across(ends_with('_id'), as.factor)) %>%  # IDs as factors
  mutate(across(timestamp, as_datetime)) # Time stamp as Date-Time

edx_clean %>%
  distinct(movie_id, genres) %>%
  count(movie_id, genres, sort = TRUE)

max(str_count(edx_clean$genres, '\\|'))
# Max number of genres is 7

# Genres are confirmed unique per film ID, pre split cleaning OK

edx_clean %>%
  distinct(movie_id, title) %>%
  count(movie_id, title, sort = TRUE)

# Titles are confirmed unique per film ID, pre split cleaning OK
edx_clean <- edx_clean %>%
  rename(c('genre' = 'genres')) %>%
  separate_wider_regex(
    # Separate Title Text from Year of Release
    cols = title,
    patterns = c(title = '.*?', release_year = '\\([:digit:]+\\)')
  ) %>%
  mutate(title = str_squish(title),
         # Clean Title Text
         release_year = as.integer(str_extract(release_year, '[:digit:]+'))) %>%  # Extract Year as Integer)
  separate_wider_delim(genre,
                       # Separate Genres
                       delim = '|',
                       names_sep = '_',
                       too_few = 'align_start') %>%
  mutate(across(starts_with('genre_'), \(x) replace_na(x, 'None'))) %>%  # Replace NAs and convert to Factors
  # Extract Date-Time features ahead of split (not a time series given provided split)
  mutate(
    review_date = as_date(timestamp),
    # Review Date
    review_year = year(review_date),
    # Review Year
    review_month = month(review_date, label = TRUE, abbr = FALSE),
    # Review Month
    review_day = day(review_date),
    # Review Day
    review_weekday = wday(review_date, label = TRUE, abbr = FALSE),
    # Review Weekday
    review_week = isoweek(review_date),
    # Review ISO week number
    review_hour = hour(timestamp),
    # Review Hour
    review_minute = minute(timestamp),
    # Review Minute
    review_second = as.integer(second(timestamp)),
    # Review Second
    review_decade = review_year - review_year %% 10,
    # Review Decade
    release_decade = release_year - release_year %% 10,
    # Release Decade
    is_am = as.factor(am(timestamp)), # Review take place in the AM
    film_age = review_year - release_year
  ) %>%
  mutate(across(ends_with('_decade'), as.integer)) # Decades as integers

# Partition Data ----------------------------------------------------------

# Define 80/20 Split Index
set.seed(1539, sample.kind = 'Rounding')
index <- createDataPartition(
  y = edx_clean$rating,
  times = 1,
  p = 0.8,
  list = FALSE
)

# Create Train and Temp Test Split
train <- edx_clean[index, ]
temp <- edx_clean[-index, ]

# Balance IDs across Train and Temp to create Test split
test <- temp %>%
  semi_join(train, by = "movie_id") %>%
  semi_join(train, by = "user_id")

# Replace observations back into train set
removed <- anti_join(temp, test)
train <- rbind(train, removed)

# Validate Balance
edx_clean %>%
  summarise(across(ends_with('_id'), n_distinct))

train %>%
  summarise(across(ends_with('_id'), n_distinct))

# IDs balanced

# Arrange Train Set in timestamp order
train <- train %>%
  arrange(timestamp)

clear_memory(keep = c(
  'train',
  'test',
  'final_holdout_test'
))

# Explore Data ------------------------------------------------------------

# Set default theme
theme_set(theme_bw())

## Rating Distribution ----------------------------------------------------

mean_rating <- mean(train$rating)
mean_rating_label <- paste('$\\bar{\\mu_{rating}}=$',round(mean_rating, 2))
rating_dist_breaks <- round(seq(0.5,5,0.5), 1)

rating_dist_plot <- train %>%
  ggplot(aes(rating)) +
  geom_density(fill = '#F8766D') +
  geom_vline(xintercept = mean_rating,
             linetype = 'dashed') +
  annotate('label',
           x = mean_rating,
           y = 0,
           label = TeX(mean_rating_label)) +
  scale_y_continuous('Count', labels = label_comma()) +
  scale_x_continuous('Rating', breaks = rating_dist_breaks)

rating_dist_plot

# Mean lies between 3 & 4
# 3.5, closest to mean is however less prominent
# Mean rating may change by some factor

clear_memory(keep = c(   'train',   'test',   'final_holdout_test', 'mean_rating', 'mean_rating_label',
                         'rating_dist_breaks'))

## Mean Rating Distribution by Time ----------------------------------------

mean_daily_rating <- train %>% 
  group_by(review_date) %>% 
  summarise(mean_rating = mean(rating))

mean_daily_rating_plot <- mean_daily_rating %>% 
  ggplot(aes(review_date, mean_rating)) +
  geom_line(color = '#F8766D') +
  geom_hline(yintercept = mean_rating,
             linetype = 'dashed') +
  geom_smooth(data = train,
              color = '#00BFC4',
              alpha = 0.25,
              aes(review_date, rating)) +
  annotate('label',
           y = mean_rating,
           x = min(mean_daily_rating$review_date),
           label = TeX(mean_rating_label)) +
  scale_x_date('Review Date',
               date_breaks = '1 year',
               labels = label_date(format = '%Y')) +
  ylab('Mean Rating')

mean_daily_rating_plot

# Rating Distribution over the entire time span does converge on the mean rating
# however there are periods in which they deviate by large margins
# Calculate distribution counts through years
# There may be a benefit in adding some sort of anchor time feature for which
# to calculate predictions in a regards of time instead of categorical features
# as year, month and day
# The smoothing line suggests that the feature should be 
# smaller than a year but larger than a day
# calculating either weeks since activity start or months

clear_memory(keep = c(   'train',   'test',   'final_holdout_test', 'mean_rating','mean_rating_label',
                         'rating_dist_breaks'))

## Rating Distribution by Year --------------------------------------------

rating_dist_year_plot <- train %>% 
  ggplot(aes(x = rating, y = as.factor(review_year))) +
  stat_density_ridges(bins = n_distinct(train$rating),
                      fill = '#F8766D',
                      quantile_lines = TRUE,
                      quantile_fun = mean
                      ) +
  geom_vline(xintercept = mean_rating,
             linetype = 'dashed') +
  annotate('label',
           x = mean_rating,
           y = 1,
           label = TeX(mean_rating_label)) +
  ylab('Review Year') +
  scale_x_continuous('Rating', breaks = rating_dist_breaks)

rating_dist_year_plot

# Half star may not have been an option before 2002
# This will influence the predicted rating
# clustering by the availability of rating options recommended

clear_memory(keep = c(   'train',   'test',   'final_holdout_test','mean_rating_label',
                         'rating_dist_breaks','mean_rating'))

## Rating Counts by Time --------------------------------------------------

rating_counts <- train %>% 
  count(review_date,rating) %>% 
  mutate(rating = as.factor(rating)) %>% 
  group_by(rating) %>% 
  mutate(rolling_count = cumsum(n)) %>% 
  ungroup() 

rating_counts_labels <- rating_counts %>% 
  group_by(rating) %>% 
  slice_max(review_date) %>% 
  ungroup()

rating_counts_plot <- rating_counts %>% 
  ggplot(aes(review_date, rolling_count, color = rating)) +
  geom_line(show.legend = FALSE,
            linewidth = 1) +
  geom_text_repel(data = rating_counts_labels,
                  show.legend = FALSE,
                  aes(review_date, rolling_count, label = as.character(rating)),
                  nudge_x = 100) +
  scale_y_continuous('Count', labels = label_comma()) +
  scale_x_date('Review Date',
               date_breaks = '1 year',
               labels = label_date(format = '%Y'))

rating_counts_plot

# Rating counts for half-stars are an option after early 2003
# 3.5 and 4.5 are the ratings which narrow the gap with medium frequency
# full ratings
# There are periods of stable review counts and periods of rapid review counts
# 3.5 and 4.5 have been seen as approaching equal proportions towards
# the end dates of the data
# review date likely to play a role in fine tuning the prediction towards an
# rating
# K-means grouping not recommended as it will result in data leakage

clear_memory(keep = c(   'train',   'test',   'final_holdout_test', 'mean_rating_label','mean_rating'))

## Time Slices ------------------------------------------------------------

# Study time slices by user intervals and filter by pair-wise correlations
times_user <- train %>% 
  group_by(user_id) %>% 
  summarise(start_date = min(review_date),
            end_date = max(review_date),
            n = n()
            )

working_cores <- detectCores() - 1

plan(multisession, workers = working_cores)

times_intervals <- future_map2(times_user$start_date, times_user$end_date,\(x,y) interval(x,y))
times_days <- future_map_dbl(times_intervals,\(x) x/days(1))

plan(sequential)

times_user <- times_user %>% 
  mutate(times_days = times_days + 1,
         days = n/times_days,
         weeks = n/times_days/52.142,
         months = n/times_days/12
         )

times_user %>% 
  summarise(across(c('days','weeks','months'),mean))

# On average users perform 36.7 reviews per day, 0.703 per week and
# 3.05 reviews per month
# slicing by day is expected to yield better results as it maximizes the
# amount of observations with weeks and months being too broad to capture
# user variations

min(train$review_date)
# Start Date 1996-01-29
max(train$review_date)
# End Date 2009-01-05

# Floor min date and ceiling ax date by year

days_n <- tibble(date = seq(floor_date(min(train$review_date), unit = 'year'),
                            ceiling_date(max(train$review_date),unit = 'year'), by = 'day')) %>% 
  mutate(days_n = row_number() - 1)

train <- train %>% 
  left_join(days_n, by = c('review_date' = 'date')) %>% 
  relocate(days_n, .after = review_date)

clear_memory(keep = c(   'train',   'test',   'final_holdout_test','mean_rating_label','mean_rating','working_cores','days_n'))

## Rating Distribution by IDs ---------------------------------------------

user_means <- train %>% 
  group_by(user_id) %>% 
  summarise(mean_rating = mean(rating)) %>% 
  rename(c('id' = 'user_id'))

user_mean_rating <- mean(user_means$mean_rating)
user_sd_rating <- sd(user_means$mean_rating)
user_mean_rating_label <- paste('$\\bar{\\mu_{User}}=$',round(user_mean_rating, 2))

film_means <- train %>% 
  group_by(movie_id) %>% 
  summarise(mean_rating = mean(rating)) %>% 
  rename(c('id' = 'movie_id'))

film_mean_rating <- mean(film_means$mean_rating)
film_sd_rating <- sd(film_means$mean_rating)
film_mean_rating_label <- paste('$\\bar{\\mu_{Film}}=$',round(film_mean_rating, 2))

all_mean <- bind_rows(list(users = user_means, films = film_means), .id = 'type') %>% 
  mutate(type = as.factor(str_to_title(type)))

all_mean_plot <- all_mean %>% 
  ggplot(aes(mean_rating, fill = type)) +
  geom_density(alpha = 0.1) +
  geom_vline(xintercept = user_mean_rating,
             color = '#00BFC4',
             linetype = 'dashed') +
  geom_vline(xintercept = film_mean_rating,
             color = '#F8766D',
             linetype = 'dashed') +
  geom_vline(xintercept = mean_rating,
             linetype = 'dashed') +
  annotate('label',
           x = user_mean_rating,
           y = 0,
           label = TeX(user_mean_rating_label)) +
  annotate('label',
           x = film_mean_rating,
           y = 0,
           label = TeX(film_mean_rating_label)) +
  annotate('label',
           x = mean_rating,
           y = 0.15,
           label = TeX(mean_rating_label)) +
  stat_function(fun = dnorm, args = list(mean = user_mean_rating, sd = user_sd_rating),
                color = '#00BFC4',
                linewidth = 1) +
  stat_function(fun = dnorm, args = list(mean = film_mean_rating, sd = film_sd_rating),
                color = '#F8766D',
                linewidth = 1) +
  xlab('Rating') +
  ylab(TeX('$Density_{scaled}$')) +
  guides(fill = guide_legend('Type'))

all_mean_plot

# User ratings have a slight negative skew
# Film ratings have a large negative skew
# neither is perfectly normal
# OLS will be expected to have an error due to the skew
# Transforming the data may aid with proper training

box_trans <- preProcess(as.data.frame(all_mean),
           method = c('center','scale','BoxCox'))

all_mean_bc <- tibble(predict(box_trans,as.data.frame(all_mean)))
all_mean_bc_mean_rating <- mean(tibble(predict(box_trans,as.data.frame(setNames(select(train,'rating'),'mean_rating'))))$mean_rating)
all_mean_bc_sd_rating <- sd(tibble(predict(box_trans,as.data.frame(setNames(select(train,'rating'),'mean_rating'))))$mean_rating)

all_mean_bc_stats <- all_mean_bc %>% 
  group_by(type) %>% 
  summarise(mean = mean(mean_rating),
            sd = sd(mean_rating))

user_mean_rating_label <- paste('$\\bar{\\mu_{User}}=$',round(all_mean_bc_stats[[2,2]], 2))
film_mean_rating_label <- paste('$\\bar{\\mu_{Film}}=$',round(all_mean_bc_stats[[1,2]], 2))
mean_rating_label_bc <- paste('$\\bar{\\mu_{rating}}=$',round(all_mean_bc_mean_rating, 2))

all_mean_bc_plot <- all_mean_bc %>% 
  ggplot(aes(mean_rating, fill = type)) +
  geom_density(alpha = 0.1) +
  geom_vline(xintercept = all_mean_bc_stats[[2,2]],
             color = '#00BFC4',
             linetype = 'dashed') +
  geom_vline(xintercept = all_mean_bc_stats[[1,2]],
             color = '#F8766D',
             linetype = 'dashed') +
  geom_vline(xintercept = all_mean_bc_mean_rating,
             linetype = 'dashed') +
  annotate('label',
           x = all_mean_bc_stats[[2,2]],
           y = 0,
           label = TeX(user_mean_rating_label)) +
  annotate('label',
           x = all_mean_bc_stats[[1,2]],
           y = 0,
           label = TeX(film_mean_rating_label)) +
  annotate('label',
           x = all_mean_bc_mean_rating,
           y = 0.05,
           label = TeX(mean_rating_label_bc)) +
  stat_function(fun = dnorm, args = list(mean = all_mean_bc_stats[[2,2]], sd = all_mean_bc_stats[[2,3]]),
                color = '#00BFC4',
                linewidth = 1) +
  stat_function(fun = dnorm, args = list(mean = all_mean_bc_stats[[1,2]], sd = all_mean_bc_stats[[1,3]]),
                color = '#F8766D',
                linewidth = 1) +
  stat_function(fun = dnorm, args = list(mean = all_mean_bc_stats[[1,2]], sd = all_mean_bc_stats[[1,3]]),
                color = '#F8766D',
                linewidth = 1) +
  stat_function(fun = dnorm, args = list(mean = all_mean_bc_mean_rating, sd = all_mean_bc_sd_rating),
                color = '#000000',
                linewidth = 1) +
  xlab(TeX('$Rating_{BoxCox}$')) +
  ylab(TeX('$Density_{scaled}$')) +
  guides(fill = guide_legend('Type'))

all_mean_bc_plot

# There is still some noticeable skew particularly with film
# a form of regularization or auxiliary variable is recommended
# a model where user effects and film effect are used is recommended
# large amount of data, an analytically solution is recommended
# OLS with L2 regularization recommended as it has an closed form solution 
# rating = film effects + user effects

clear_memory(keep = c(   'train',   'test',   'final_holdout_test', 'mean_rating','working_cores','days_n'))

### Matrix Sparsity -------------------------------------------------------

user_sparcity <- train %>% 
  count(user_id) %>% 
  mutate(user_sparsity = n/length(unique(train$movie_id)))

mean_user_sparcity <- mean(user_sparcity$user_sparsity)
mean_user_sparcity_label <- paste('$User\\,Sparcity=$',round(mean_user_sparcity, 2))

spacity_plot <- user_sparcity %>% 
  ggplot(aes(user_sparsity)) +
  geom_density(fill = '#F8766D') +
  geom_vline(xintercept = mean(user_sparcity$user_sparsity),
             linetype = 'dashed') +
  annotate('label',
           x = mean_user_sparcity,
           y = 10,
           label = TeX(mean_user_sparcity_label)) +
  geom_vline(xintercept = 1) +
  annotate('label',
           x = 1,
           y = 10,
           label = TeX('$OLS_{Tipical\\,Sparsity}$')) +
  xlab('User Sparcity') +
  ylab('Density')

spacity_plot

# or typical OLS each user should have at a minimum 1 review per film
# the mean user sparsity is 0.01 and the min requirement would be 1 for
# a min OLS
# users and films will have different effects on prediction
# user effect + film effects further validated

clear_memory(keep = c(   'train',   'test',   'final_holdout_test', 'mean_rating','working_cores','days_n'))

## Genre Effects ----------------------------------------------------------

# There are a total of 8 genre levels however some are labeled as "none"

film_genres <- train %>% 
  distinct(movie_id, .keep_all = TRUE) %>% 
  select(c(movie_id, starts_with('genre_')))

film_genre_tidy <- film_genres %>%
  pivot_longer(
    cols = starts_with('genre_'),
    names_to = 'genre_level',
    values_to = 'genre',
    names_transform = list(genre_level = ~ as_factor(str_remove(.x, 'genre_')))
  )

film_genre_tidy %>% 
  group_by(genre_level) %>% 
  add_count(name = 'level_n') %>% 
  group_by(genre_level, genre) %>% 
  summarise(frequency = n()/first(level_n)) %>% 
  ungroup() %>% 
  filter(genre == 'None') %>% 
  filter(frequency <= 0.50)

# Using a 50/50 cutoff frequency ensuring at least 1/2 of a variable is not "None"
# only Levels 1-2 have sufficient frequency of 
# none "None" genres to be usable for adequate clustering analysis

train %>% 
  summarise(across(starts_with('genre_'), \(x) toString(unique(x)))) %>% 
  pivot_longer(everything()) %>% 
  separate_longer_delim(cols = 'value',
                        delim = ', ') %>% 
  distinct(value) %>% 
  pull(value)

# Replace "(no genres listed)" with none

film_genres <- film_genres %>% 
  mutate(across(starts_with('genre_'),\(x) str_replace(x,'\\([[:alpha:][:space:]]+\\)','None')))

film_genres <- film_genres %>% 
  select(-all_of(str_c('genre',3:8, sep = '_')))

genre_means <- train %>% 
  mutate(across(starts_with('genre_'),\(x) str_replace(x,'\\([[:alpha:][:space:]]+\\)','None'))) %>% 
  group_by(genre_1,genre_2) %>% 
  summarise(mean_rating = mean(rating)) %>% 
  ungroup() %>% 
  mutate(label = label_number(accuracy = 0.01)(mean_rating))

genre_means_heatmap <- genre_means %>% 
  ggplot(aes(genre_1, genre_2, fill = mean_rating, label = label)) +
  geom_tile() +
  geom_text(color = '#FFFFFF') +
  scale_fill_continuous('Rating',high = "#132B43", low = "#56B1F7") +
  xlab('Genre 1') +
  ylab('Genre 2')

genre_means_heatmap

# Genre combinations demonstrate a clear difference in mean ratings which may 
# aid in prediction for user effects, film effects excluded as genres
# are inherent properties of a film
# to minimize features genres clustering to be explored via K-Modes Clustering
# specifically using the weighted distance to account for category imbalances 

genre_levels <- unique(film_genres$genre_1)

genre_max <- film_genres[,-1] %>% 
  distinct() %>% 
  nrow()

film_genres_num <- film_genres %>% 
  mutate(across(starts_with('genre_'), \(x) as.numeric(factor(x,levels = genre_levels))))

kmodes_withindiff <- function(data, modes, seed = 1443){
  
  suppressWarnings(set.seed(seed = seed, sample.kind = 'Rounding'))
  withindiff <- sum(kmodes(data, modes = modes, iter.max = 10, weighted = TRUE, fast = TRUE)$withindiff)
  
  return(withindiff)
  
}

plan(multisession, workers = working_cores)

genre_withindiff <- tibble(genres = 1:genre_max,
                           withindiff = future_map_dbl(genres,\(x) kmodes_withindiff(film_genres_num[,-1],x),
                                                .progress = TRUE)
                           )

plan(sequential)

genre_withindiff_plot <- genre_withindiff %>% 
  ggplot(aes(genres, withindiff)) +
  geom_point() +
  geom_line() +
  geom_smooth(se = FALSE, method = 'lm') +
  geom_text_repel(aes(label = as.character(genres))) +
  xlab('Genres') +
  ylab('Weighted within-cluster distance')

genre_withindiff_plot

# Typically 22 would be chosen as the optimal amount of genres
# However the amount of variation captured by these is not sufficient
# the optimal genre clusters will determined by the a 80% variance threshold

genres_clusters <- genre_withindiff %>% 
  mutate(prop = withindiff/sum(withindiff),
         c_prop = cumsum(prop)) %>% 
  filter(c_prop <= 0.8) %>% 
  slice_max(c_prop) %>% 
  pull(genres)

genres_clusters

# 57 Genres considered optimal at a 80% variance threshold

set.seed(seed = 1443, sample.kind = 'Rounding')
genre_model <- kmodes(film_genres_num[,-1], modes = genres_clusters,
                      weighted = TRUE)

film_genres <- film_genres %>% 
  mutate(genre = as.factor(genre_model$cluster))

film_cluster_plot <- film_genres %>% 
  pivot_longer(cols = starts_with('genre_'),
               names_to = 'level',
               values_to = 'genre_name',
               names_transform = \(x) as.integer(str_extract(x,'[:digit:]+$'))) %>% 
  group_by(genre, genre_name) %>% 
  summarise(level = toString(sort(unique(level)))) %>% 
  ungroup() %>% 
  ggplot(aes(genre, genre_name, fill = genre, label = as.character(level))) +
  geom_tile(show.legend = FALSE) +
  geom_text() +
  xlab('Genre Cluster') +
  ylab('Genre')

film_cluster_plot

train <- train %>% 
  mutate(across(starts_with('genre_'),\(x) str_replace(x,'\\([[:alpha:][:space:]]+\\)','None'))) %>% 
  left_join(select(film_genres,-all_of(str_c('genre',1:2,sep = '_'))),
            by = 'movie_id') %>%
  relocate(genre, .before = 'genre_1') %>% 
  select(-starts_with('genre_'))

clear_memory(keep = c(   'train',   'test',   'final_holdout_test', 'film_genres','working_cores','days_n'))

# Explicit Feature Selection ----------------------------------------------

train <- train %>% 
  relocate(rating) %>% 
  select(-all_of(c('timestamp','title','review_date')))

## Near Zero Variance -----------------------------------------------------

nzv_names <- nearZeroVar(train[,-1],
            names = TRUE,
            saveMetrics = TRUE,
            foreach = TRUE,
            allowParallel = TRUE)

which(nzv_names$nzv)

# There are no near zero variance variables

## Correlated Predictors --------------------------------------------------

train_num <- train %>% 
  mutate(across(where(is.ordered),as.numeric)) %>% 
  select(where(is.numeric) & -all_of('rating'))

train_cor <- cor(train_num)
# train_cor_pmat <- cor_pmat(train_num)

numeric_corr_plot <- ggcorrplot(train_cor,
           type = 'lower',
           ggtheme = ggplot2::theme_bw(),
           hc.order = TRUE,
           lab = TRUE#,
           # p.mat = train_cor_pmat
           )

numeric_corr_plot

cor_remove <- findCorrelation(train_cor, verbose = TRUE, names = TRUE)

tibble(predictor = names(train_num),
       remove = predictor %in% cor_remove) %>% 
  arrange(desc(predictor)) %>% 
  mutate(type = case_when(str_detect(predictor,'_square$') ~ 'square',
                          str_detect(predictor,'_sqrt$') ~ 'square_root',
                          str_detect(predictor,'_cube$') ~ 'cube',
                          str_detect(predictor,'log$') ~ 'log',
                          str_detect(predictor,'log2$') ~ 'log2',
                          str_detect(predictor,'log10$') ~ 'log10',
                          str_detect(predictor,'log1p$') ~ 'log1p',
                          TRUE ~ 'original'
                          ),
         predictor = str_extract(predictor,'^[:alpha:]+\\_[:alpha:]+')) %>% 
  group_by(predictor) %>% 
  filter(remove == FALSE)
  
# swap film_age for release_year, release year is static across time
# prefer non static features as f(t) is currently considered

# cor_remove[which(cor_remove %in% 'film_age')] <- 'release_year'

train <- train %>% 
  select(-all_of(cor_remove))

clear_memory(keep = c(   'train',   'test',   'final_holdout_test', 'film_genres','working_cores','days_n'))

## Categorical Correlations -----------------------------------------------

cramerv_ass <- function(data, verbose = FALSE, bias.correct = TRUE){
  
  # Keep Nominal Categories
  data <- data %>% 
    select(where(is.factor) & !where(is.ordered))
  
  # Get Data names
  data_names <- data %>% 
    names()
  
  # Prepare empty assosiation matrix
  cramer_v <- matrix(nrow = ncol(data), ncol = ncol(data))
  
  colnames(cramer_v) <- data_names
  rownames(cramer_v) <- data_names
  
  for(i in 1:ncol(data)){
    for(n in 1:ncol(data)){
      
      if(i==n){
        
        cramer_v[i,n] <- 1
        
      }
      else{
        if((i==1 & n==2)|(i==2 & n==1)){
          
          # Film-User interaction not considered for model
          
          cramer_v[i,n] <- 0
          
        }
        else{
          
          cramer_v[i,n] <- cramerV(x = data[,i][[1]], data[,n][[1]], verbose = verbose,
                                   bias.correct = bias.correct)
          
        }
      }
    }
  }
  
  return(cramer_v)
  
}

train_cat_corr <- cramerv_ass(train)
train_cat_corr

categorical_corr_plot <- ggcorrplot(train_cat_corr,
                                    type = 'lower',
                                    ggtheme = ggplot2::theme_bw(),
                                    hc.order = TRUE,
                                    lab = TRUE,
                                    digits = 3) +
  scale_fill_gradient2(limit = c(0,1),
                       low = '#0000FF',
                       high = '#FF0000',
                       mid = '#FFFFFF',
                       midpoint = 0.4)

categorical_corr_plot

cat_cor_remove <- findCorrelation(train_cat_corr, verbose = TRUE, names = TRUE,
                cutoff = 0.6)

cat_cor_remove

# the bias corrected Cramer's V using a 0.6 cutoff would remove user and genre
# is_am is to be removed in place of user and genre will remain but only
# for training user effects as the association is very weak when paired with
# user

cat_cor_remove <- 'is_am'

train <- train %>% 
  select(-all_of(cat_cor_remove))

clear_memory(keep = c(   'train',   'test',   'final_holdout_test', 'film_genres','working_cores','days_n'))

## Linear Combos ----------------------------------------------------------

train_num <- train %>% 
  mutate(across(where(is.ordered),as.numeric)) %>% 
  select(where(is.numeric) & -all_of('rating'))

train_lc <- findLinearCombos(train_num)

train_lc

# No Linear Combos

clear_memory(keep = c(   'train',   'test',   'final_holdout_test', 'film_genres','working_cores','days_n'))

## Boruta -----------------------------------------------------------------

# Boruta must be separated by effects 
# Due to size of dataset use getImpXgboost

### User Boruta -----------------------------------------------------------

set.seed(2148, sample.kind = 'Rounding')
user_boruta <- Boruta(rating~.,
                      data = select(train,
                                    -all_of(c('movie_id'))),
                      getImp = getImpXgboost,
                      maxRuns = 50,
                      doTrace = 3)

plot(user_boruta)

# Genre and release year are the most important features for users 
# while days_n is a disant thrid

user_predictors <- getSelectedAttributes(user_boruta)[c(1:3)]

user_predictors <- c('user_id',user_predictors)

# clear_memory(keep = c(   'train',   'test',   'final_holdout_test', 'film_genres',
#                          'user_predictors','working_cores'))

### Film Boruta -----------------------------------------------------------

set.seed(2148, sample.kind = 'Rounding')
film_boruta <- Boruta(rating~.,
                      data = select(train,
                                    -all_of(c('user_id','genre','release_year'))),
                      getImp = getImpXgboost,
                      maxRuns = 50,
                      doTrace = 3)

plot(film_boruta)

# film_id is nearly the only important feature
# confirm via varrank

film_predictors <- getSelectedAttributes(film_boruta)[1:2]

## Varrank ----------------------------------------------------------------

### User Varrank ----------------------------------------------------------

user_varrank <- varrank(select(train,all_of(c('rating',user_predictors))),
        method = 'estevez',
        variable.important = 'rating',
        discretization.method = 'sturges',
        algorithm = 'forward',
        scheme = 'mid',
        verbose = TRUE
        )

summary(user_varrank)

plot(user_varrank)

# genre may become redundant when user_id is already trained
# This mostly matched the inverse of the boruta wrapper
# release year is the least relevant feature
# will proceede with genre and days_n

user_predictors <- user_varrank$ordered.var[1:3]

### Film Varrank ----------------------------------------------------------

film_varrank <- varrank(select(train,all_of(c('rating',film_predictors))),
                        method = 'estevez',
                        variable.important = 'rating',
                        discretization.method = 'sturges',
                        algorithm = 'forward',
                        scheme = 'mid',
                        verbose = TRUE)

summary(film_varrank)

plot(film_varrank)

# The wrapper inverts the findings of boruta ion terms of importance
# will onl;y train film in this instance

film_predictors <- 'movie_id'

## Final Training Data ----------------------------------------------------

train <- train %>%
  select(all_of(unique(c('rating',user_predictors, film_predictors))))

clear_memory(keep = c(   'train',   'test',   'final_holdout_test','film_genres', 'working_cores','days_n'))

# Data Preparation --------------------------------------------------------

preprocess_rating_model <- preProcess(as.data.frame(train[,1]),
                               method = c('center','scale','BoxCox'),
                               verbose = TRUE)

preprocess_days_n_model <- preProcess(train[,4],
           method = 'range',
           verbose = TRUE)

train_0 <- as_tibble(predict(preprocess_rating_model,as.data.frame(train))) %>% 
  predict(preprocess_days_n_model,.)

test %>% 
  left_join(film_genres,
            by = 'movie_id') %>% 
  left_join(days_n) %>% 
  select(all_of(names(train_0)))

# Train Model -------------------------------------------------------------

rmse_calc <- function(data){
  
  rmses <- data %>% 
    summarise(across(starts_with('y_hat'),\(x) RMSE(rating,x))) %>% 
    pivot_longer(cols = everything(),
                 names_to = 'model',
                 values_to = 'rmse',
                 names_transform = \(x) as.factor(str_to_title(str_remove(x,'y_hat_')))) %>% 
    mutate(model = fct_reorder(model,rmse))
  
  return(rmses)
  
}

rmse_plot <- function(data, label_accuracy = 0.001){
  
  plot <- data %>% 
    ggplot(aes(rmse, model, label = label_number(accuracy = label_accuracy)(rmse))) +
    geom_segment(aes(xend = 0, yend = model)) +
    geom_point() +
    geom_text(nudge_x = 0.025) +
    xlab('RMSE') +
    ylab('Model')
  
  return(plot)
  
}

sgd_lambda <- function(x, g_x, beta, y, lambda_initial = 0, learning_rate = 0.5 , epochs = 2000, seed = 1142){
  
  # Model
  lambda_model <- function(x, g_x, lambda, beta){
    return(g_x/(g_x + lambda) * beta * x)
    
  }
  
  # Presets
  n <- length(beta)
  lambda <- lambda_initial
  loss_table <- tibble(lambda = numeric(),
                       rmse = numeric())
  
  for (epoch in 1:epochs) {
    for (i in n) {
      
      # Random Data Sample
      set.seed(seed, sample.kind = 'Rounding')
      index <- sample(1:n, 1)
      beta_i <- beta[index]
      g_x_i <- g_x[index]
      x_i <- x[index]
      lambda_i <- lambda[index]
      y_i <- y[index]
      
      # Compute Gradients
      y_pred <- lambda_model(x_i, g_x_i,lambda,beta_i)
      grad_lambda <- -2 * beta_i * x_i * (y_i - y_pred)
      
      # Current RMSE
      rmse <- RMSE(y_pred, y)
      
      # Update Loss Table
      loss_table <- loss_table %>% 
        add_row(lambda = lambda,
                rmse = rmse)
      
      # Update Model Parameters
      lambda <- lambda + learning_rate * grad_lambda
      
    }
  }
  
  lambda_min <- loss_table %>% 
    slice_min(rmse, with_ties = FALSE) %>% 
    pull(lambda)
  
  return(lambda_min)
  
}

## Intercept --------------------------------------------------------------

# When data is centered and scaled the model intercept is 0 due to 
# OLS closed form solution being the mean

b0 <- 0

test %>% 
  mutate(b0 = b0) %>% 
  mutate(y_hat_intercept = b0) %>% 
  rmse_calc() %>% 
  rmse_plot()
  
train_betas <- function(data){
  
  data <-   data %>% 
    mutate(b0 = b0)
  
  return(data)
  
}

# RMSE with Intercept 1.000

## Film Effects -----------------------------------------------------------

b_film <- train_0 %>% 
  group_by(movie_id) %>% 
  mutate(b_i = rating - b0) %>% 
  summarise(b_film_xy = sum(b_i),
            b_film_gx = sum(!is.na(b_i))) %>% 
  ungroup()

train_betas <- function(data){
  
  data <-   data %>% 
    mutate(b0 = b0) %>% 
    left_join(b_film)
  
  return(data)
  
}

train_0 %>% 
  train_betas() %>% 
  mutate(y_hat_intercept = b0,
         y_hat_film = b0 + b_film_xy/b_film_gx) %>% 
  rmse_calc() %>% 
  rmse_plot()

# RMSE with film effects is 0.889  


## Penalized Film Effects -------------------------------------------------

film_ids <- 1:length(unique(as.integer(train_0$movie_id)))

set.seed(2134, sample.kind = 'Rounding')
film_index <- createMultiFolds(film_ids)

film_best_lambda <- train_0 %>% 
  train_betas() %>%
  nest() %>% 
  expand_grid(film_index) %>% 
  mutate(new_data = map2(data, film_index, \(x,y) filter(x, movie_id %in% unlist(y)),
                         .progress = TRUE)) %>% 
  mutate(lambda = map_dbl(new_data,\(x) sgd_lambda(x = rep(1,nrow(x)),
                                               y = pluck(x,'new_data',1)$rating - pluck(x,'new_data',1)$b0,
                                               g_x = pluck(x,'new_data',1)$b_film_gx,
                                               beta = pluck(x,'new_data',1)$b_film_xy/pluck(x,'new_data',1)$b_film_gx
  ), .progress = TRUE)) %>% 
  summarise(best_lambda = mean(lambda, na.rm = TRUE)) %>% 
  pull(best_lambda)

film_best_lambda

# Best Lambda is 0
# No penalization required

b_film <- b_film %>% 
  mutate(b1 = b_film_xy/b_film_gx) %>% 
  select(all_of(c('movie_id','b1')))

train_0 %>% 
  train_betas() %>% 
  mutate(y_hat_intercept = b0,
         y_hat_film = b0 + b1) %>% 
  rmse_calc() %>% 
  rmse_plot()

## Film Time Effects ------------------------------------------------------

b_film_days <- train_0 %>% 
  train_betas() %>% 
  mutate(b_i = rating - b0 - b1) %>% 
  group_by(movie_id) %>% 
  summarise(b_film_xy = sum(b_i*days_n),
            b_film_gx = sum(days_n^2))

train_0 %>% 
  train_betas() %>% 
  left_join(b_film_days) %>% 
  mutate(y_hat_intercept = b0,
         y_hat_film = b0 + b1,
         y_hat_film_days = b0 + b1 + b_film_xy/b_film_gx
         ) %>% 
  rmse_calc() %>% 
  rmse_plot()

# RMSE of Film-Days is 0.895
# An increase from the previous RMSE and a candidate for dropping
#  Explore Penalized effects

## Penalized Film Time Effects --------------------------------------------

film_time_best_lambda <- train_0 %>% 
  train_betas() %>% 
  left_join(b_film_days) %>% 
  nest() %>% 
  expand_grid(film_index) %>% 
  mutate(new_data = map2(data, film_index, \(x,y) filter(x, movie_id %in% unlist(y)),
                         .progress = TRUE)) %>% 
  mutate(lambda = map_dbl(new_data,\(x) sgd_lambda(x = pluck(x,'new_data',1)$b_film_xy,
                                                   y = pluck(x,'new_data',1)$rating - pluck(x,'new_data',1)$b0 - pluck(x,'new_data',1)$b1,
                                                   g_x = pluck(x,'new_data',1)$b_film_gx,
                                                   beta = pluck(x,'new_data',1)$b_film_xy/pluck(x,'new_data',1)$b_film_gx
  ), .progress = TRUE)) %>% 
  summarise(best_lambda = mean(lambda, na.rm = TRUE)) %>% 
  pull(best_lambda)

film_time_best_lambda

# Best Lambda is 0
# Penalization did not produce a better model
# Film-time not added to model

## User Effects -----------------------------------------------------------

b_user <- train_0 %>% 
  train_betas() %>% 
  mutate(b_i = rating - b0 - b1) %>% 
  group_by(user_id) %>% 
  summarise(b_user_xy = sum(b_i),
            b_user_gx = sum(!is.na(b_i))) %>% 
  ungroup()

train_0 %>% 
  train_betas() %>% 
  left_join(b_user) %>% 
  mutate(y_hat_intercept = b0,
         y_hat_film = b0 + b1,
         y_hat_user = b0 + b1 + b_user_xy/b_user_gx
  ) %>% 
  rmse_calc() %>% 
  rmse_plot()

# RMSE with user effects is 0.807
# Model improved
# Study potential penalization for an improved model

train_betas <- function(data){
  
  data <-   data %>% 
    mutate(b0 = b0) %>% 
    left_join(b_film) %>% 
    left_join(b_user)
  
  return(data)
  
}

## Penalized User Effects -------------------------------------------------

user_ids <- 1:length(unique(as.integer(train_0$user_id)))

set.seed(2134, sample.kind = 'Rounding')
user_index <- createMultiFolds(user_ids)

user_best_lambda <- train_0 %>% 
  train_betas() %>%
  nest() %>% 
  expand_grid(user_index) %>% 
  mutate(new_data = map2(data, user_index, \(x,y) filter(x, user_id %in% unlist(y)),
                         .progress = TRUE)) %>% 
  mutate(lambda = map_dbl(new_data,\(x) sgd_lambda(x = rep(1,nrow(x)),
                                                   y = pluck(x,'new_data',1)$rating - pluck(x,'new_data',1)$b0 - pluck(x,'new_data',1)$b1,
                                                   g_x = pluck(x,'new_data',1)$b_user_gx,
                                                   beta = pluck(x,'new_data',1)$b_user_xy/pluck(x,'new_data',1)$b_user_gx
  ), .progress = TRUE)) %>% 
  summarise(best_lambda = mean(lambda, na.rm = TRUE)) %>% 
  pull(best_lambda)

user_best_lambda

# Best Lambda is 0
# No penalization required

b_user <- b_user %>% 
  mutate(b2 = b_user_xy/b_user_gx) %>% 
  select(all_of(c('user_id','b2')))

train_0 %>% 
  train_betas() %>% 
  mutate(y_hat_intercept = b0,
         y_hat_film = b0 + b1,
         y_hat_user = b0 + b1 + b2) %>% 
  rmse_calc() %>% 
  rmse_plot()

## User Time Effects ------------------------------------------------------

b_user_days <- train_0 %>% 
  train_betas() %>% 
  mutate(b_i = rating - b0 - b1 - b2) %>% 
  group_by(user_id) %>% 
  summarise(b_user_xy = sum(b_i*days_n),
            b_user_gx = sum(days_n^2))

train_0 %>% 
  train_betas() %>% 
  left_join(b_user_days) %>% 
  mutate(y_hat_intercept = b0,
         y_hat_film = b0 + b1,
         y_hat_user = b0 + b1 + b2,
         y_hat_user_days = b0 + b1 + b2 + b_user_xy/b_user_gx
  ) %>% 
  rmse_calc() %>% 
  rmse_plot()

# RMSE of User-Days is 0.973
# An increase from the previous RMSE and a candidate for dropping
# Explore Penalized effects

## Penalized User Time Effects --------------------------------------------

user_time_best_lambda <- train_0 %>% 
  train_betas() %>% 
  left_join(b_user_days) %>% 
  nest() %>% 
  expand_grid(user_index) %>% 
  mutate(new_data = map2(data, user_index, \(x,y) filter(x, movie_id %in% unlist(y)),
                         .progress = TRUE)) %>% 
  mutate(lambda = map_dbl(new_data,\(x) sgd_lambda(x = pluck(x,'new_data',1)$b_user_xy,
                                                   y = pluck(x,'new_data',1)$rating - pluck(x,'new_data',1)$b0 - pluck(x,'new_data',1)$b1 - pluck(x,'new_data',1)$b2,
                                                   g_x = pluck(x,'new_data',1)$b_user_gx,
                                                   beta = pluck(x,'new_data',1)$b_user_xy/pluck(x,'new_data',1)$b_user_gx
  ), .progress = TRUE)) %>% 
  summarise(best_lambda = mean(lambda, na.rm = TRUE)) %>% 
  pull(best_lambda)

user_time_best_lambda

# Best Lambda is 0
# Penalization did not produce a better model
# User-time not added to model

# Timer -------------------------------------------------------------------

tictoc::toc()
