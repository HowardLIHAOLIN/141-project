---
title: "STA 141A project"
author: "HAOLIN LI"
date: "2025-03-06"
output:
  html_document: default
  pdf_document: default
---
```{r}
Sys.setenv(LANG = "en_US.UTF-8")
Sys.setlocale("LC_ALL", "en_US.UTF-8")
```
# Libarary
```{r}
library(readr)
library(dplyr)
library(ggplot2)
library(tibble)
library(tidyr)
library(reshape2)
library(keras)
library(abind)
library(rmarkdown)



```

# Instruction
This project is based on the research results of Steinmetz et al. (2019) and aims to explore the relationship between neuronal spike activity and behavioral outcomes in visual decision-making tasks and build a prediction model to predict the feedback type (success or failure) of the experiment. In this study, a total of 10 mice participated in the experiment in 39 sessions, each of which contained hundreds of trials. During the experiment, the mice were randomly presented with visual stimuli of different contrasts (values {0, 0.25, 0.5, 1}, where 0 means no stimulation) on the screens on both sides, and used their forepaws to operate the steering wheel to make decisions. Depending on the decision results, the mice will receive rewards (success, feedback type 1) or be punished (failure, feedback type -1). The specific rules are as follows:

When the contrast on the left side is greater than that on the right side, if the mouse turns right, it is a success, otherwise it is a failure.

When the contrast on the right side is greater than that on the left side, if the mouse turns left, it is a success, otherwise it is a failure.

When there is no stimulation on both sides, if the mouse remains still, it is a success, otherwise it is a failure.

When the left-right contrast is equal but non-zero, the left-right turn is randomly selected (50% correct rate).

In addition, the activity of neurons in the visual cortex of mice was recorded during the experiment. The data is provided in the form of spike trains, that is, the discharge timestamps of neurons are recorded from the start of the trial to 0.4 seconds (40 time steps). This project only selects 18 sessions from four mice (Cori, Frossman, Hence, Lederberg) for analysis.

In this project, we will first perform exploratory data analysis to understand the data structure, the activity patterns of different brain regions in each session, and the distribution of experimental results. Subsequently, we will extract key statistical features reflecting neural activity in each trial (e.g., the average discharge rate and standard deviation of different brain regions at each time step) through feature engineering and data integration, and construct multi-dimensional inputs combined with visual stimulus information. Finally, we plan to use convolutional neural networks (CNNs) to build predictive models to explore the complex relationship between neural activity patterns and experimental success rates.


# Data Structure
## describe the data structures across sessions
```{r}
data <- list()
for (i in 1:18) {
  file_path <- sprintf("./Data/session%d.rds", i)
  data[[i]] <- readRDS(file_path)
  print(data[[i]]$mouse_name)
  print(data[[i]]$date_exp)
}

```
Variables names in sessions. 
This helpes to get an idea of what variables are in the data.

## Showing the variable names of the data
```{r}
names(data[[1]])
```
Spike Data. 
This helps to understand how the spikes collected in the data
```{r}
session[[1]]$spks[[1]][6,] 

```

# Data processing
## Summery of the data structure
```{r}
# Function to summarize data from each session
summarize_session <- function(session) {
  list(
    Number_of_Neurons = ifelse(is.list(session$spks), nrow(session$spks[[1]]), NA), 
    Brain_Area_Count = if (!is.null(session$brain_area)) length(session$brain_area) else NA,
    Number_of_Time_Bins = ncol(session$spks[[1]])

    
  )
}

# Apply the summarizing function to each session
session_summaries <- lapply(data, summarize_session)
session_summary_df <- do.call(rbind, session_summaries)
print(session_summary_df)


```
It shows that the data is supper large so it need dimensionality reduction
From this table we can see that not all sessions have the same brain area.

# EDA
## Number of brain area
```{r}
brain_area_counts <- sapply(data, function(session) {
  length(unique(session$brain_area))
})


print(brain_area_counts)
```
### Visulization
```{r}

brain_areas_per_session <- lapply(session, function(s) {
  unique(s$brain_area)
})

brain_area_df <- data.frame(
  session = rep(1:18, sapply(brain_areas_per_session, length)),
  brain_area = unlist(brain_areas_per_session),
  stringsAsFactors = FALSE
)


ggplot(brain_area_df, aes(x = factor(session), fill = brain_area)) +
  geom_bar() + 
  labs(x = "Session", y = "Count of Brain Areas", title = "Brain Areas Recorded in Each Session") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) 

```
From the data and graph it shows that the number of brain regions varied widely between sessions, ranging from a low of 5 to a high of 15. This suggests that there may be significant differences in experimental setup or data collection between sessions. Such differences between sessions can have an impact on data analysis and model training. For example, when performing cross-session comparisons or aggregate analysis, potential biases that may be caused by these differences need to be taken into account.Therefore, dimensionality reduction is required.



## Visual stimulation information have a relationship with success

```{r}
trial_data <- do.call(rbind, lapply(seq_along(data), function(i) {
  s <- data[[i]]
  n <- length(s$spks)  
  data.frame(
    session_id = i,
    trial_id = 1:n,
    contrast_left = s$contrast_left,
    contrast_right = s$contrast_right,
    feedback_type = s$feedback_type  
  )
}))

trial_data <- trial_data %>%
  mutate(success = ifelse(feedback_type == 1, 1, 0))

trial_data <- trial_data %>%
  mutate(
    contrast_condition = case_when(
      contrast_left > contrast_right ~ "Left > Right",
      contrast_right > contrast_left ~ "Right > Left",
      contrast_left == 0 & contrast_right == 0 ~ "No Stimulus",
      contrast_left == contrast_right & contrast_left > 0 ~ "Equal Nonzero"
    )
  )

condition_summary <- trial_data %>%
  group_by(contrast_condition) %>%
  summarise(
    num_trials = n(),
    success_rate = mean(success, na.rm = TRUE)
  ) %>%
  ungroup()

print(condition_summary)

ggplot(condition_summary, aes(x = contrast_condition, y = success_rate, fill = contrast_condition)) +
  geom_bar(stat = "identity") +
  labs(title = "Success rate under different visual stimulation conditions",
       x = "Visual stimulation conditions",
       y = "Success rate") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


```
The differences in success rates under different conditions indicate that visual stimulation does have an impact on the mice??s decision-making process, which also affects the final accuracy rate. This result further supports the view that the more obvious the difference in visual stimulation, the more accurate the mice??s decision-making.


## Succesful rate change over time 

```{r}
trial_data <- do.call(rbind, lapply(seq_along(data), function(s_idx) {
  s <- data[[s_idx]]
  n_trials <- length(s$spks)
  
  all_spike_rates <- lapply(seq_len(n_trials), function(trial_id) {
    spikes <- s$spks[[trial_id]]
    colMeans(spikes)
  })
  
  df_session <- do.call(rbind, lapply(seq_len(n_trials), function(trial_id) {
    data.frame(
      session_id = s_idx,
      trial_id = trial_id,
      time_bin = 1:length(all_spike_rates[[trial_id]]),
      spike_rate = all_spike_rates[[trial_id]],
      success = ifelse(s$feedback_type[trial_id] == 1, 1, 0)
    )
  }))
  
  return(df_session)
}))

time_bin_summary <- trial_data %>%
  group_by(time_bin, success) %>%
  summarise(
    mean_spike_rate = mean(spike_rate, na.rm = TRUE),
    .groups = "drop"
  )

ggplot(time_bin_summary, aes(x = time_bin, y = mean_spike_rate, color = factor(success))) +
  geom_line() +
  labs(
    title = "Spike Rate over Time Bins: Success vs. Failure",
    x = "Time Bin",
    y = "Average Spike Rate",
    color = "Success"
  ) +
  theme_minimal()
```
In most time periods, the blue curve representing successful trials is significantly higher than the red curve. The difference is particularly obvious in some time bins (e.g., from about the 5th to the 20th time slice in the figure), which may indicate that the increase in success rate is accompanied by higher neural activity over time.

## Standard Deviation of Spike Rate over Time

```{r}
library(dplyr)
library(ggplot2)

time_bin_sd <- trial_data %>%
  group_by(time_bin, success) %>%
  summarise(
    sd_spike_rate = sd(spike_rate, na.rm = TRUE),  # ??准??
    .groups = "drop"
  )


ggplot(time_bin_sd, aes(x = time_bin, y = sd_spike_rate, color = factor(success))) +
  geom_line() +
  labs(
    title = "Standard Deviation of Spike Rate over Time Bins: Success vs. Failure",
    x = "Time Bin",
    y = "Spike Rate SD",
    color = "Success"
  ) +
  theme_minimal()

```
In the figure, the gap between the success group and the failure group is particularly large between the 5th and 20th time bins, indicating that the variability (SD) of neural activity during this time period can best distinguish success from failure. Higher variability (SD) may mean that the animal's response pattern to the stimulus is more diverse, and this diversity may be associated with successful decision-making during the critical time period.


## Average spike rate of each brain area of sessions
```{r}
calculate_averages <- function(session) {
  spikes <- session$spks  
  num_trials <- length(spikes)
  brain_areas <- unique(session$brain_area)
  time_bins <- ncol(session$spks[[1]])  

  avg_data <- list()

  for (area in brain_areas) {
    area_indices <- which(session$brain_area == area)
    
    area_spikes <- lapply(spikes, function(mat) {
      if (ncol(mat) > 0 && !is.null(mat) && length(area_indices) > 0) {
        return(mat[area_indices, , drop = FALSE])
      }
    })

    area_spikes <- Filter(function(x) !is.null(x) && nrow(x) > 0, area_spikes)

    if (length(area_spikes) > 0) {
      combined_spikes <- do.call(abind, c(area_spikes, along = 3))
      area_means <- apply(combined_spikes, 2, function(x) mean(x, na.rm = TRUE))

      if (length(area_means) == time_bins) {
        avg_data[[area]] <- data.frame(
          time_bin = 1:time_bins,
          avg_spike_rate = area_means
        )
      }
    }
  }

  return(avg_data)
}


```


### Visualize the data
```{r}
plot_area_averages <- function(avg_data) {
  plots <- lapply(names(avg_data), function(area) {
    data <- avg_data[[area]]
    ggplot(data, aes(x = time_bin, y = avg_spike_rate)) +
      geom_line() +
      ggtitle(paste("Average Spike Rate in", area)) +
      xlab("Time Bin") +
      ylab("Average Spike Rate")
  })
  return(plots)
}

session_averages <- lapply(data, calculate_averages)

all_sessions_data <- do.call(rbind, lapply(seq_along(session_averages), function(i) {
  session_data <- bind_rows(session_averages[[i]], .id = "brain_area")
  session_data$session_id <- i
  return(session_data)
}))

all_sessions_data <- all_sessions_data %>%
  mutate(brain_area = as.factor(brain_area),
         session_id = factor(session_id),
         time_bin = as.numeric(time_bin),
         avg_spike_rate = as.numeric(avg_spike_rate))

p <- ggplot(all_sessions_data, aes(x = time_bin, y = avg_spike_rate, color = brain_area)) +
  geom_line() +
  labs(title = "Average Spike Rate Across Sessions",
       x = "Time Bin",
       y = "Average Spike Rate") +
  facet_wrap(~session_id, scales = "free_y") +
  theme_minimal() +
  theme(legend.position = "bottom")

print(p)



```
The curves in the figure show that the average spike rate in certain brain regions increases or decreases significantly during a specific time period. For example graph 12 This indicates that brain activity is closely related to time.





# Prediction

From all the above data analysis, it can be seen that all variables are related to time and have a positive relationship with the success rate. Considering the high dimensionality of the data and the possible interactions between the variables, I decided to use a time series model, CNN, to build a prediction model and capture these dynamic patterns.



## Data preprocessing
Before building the model, the data dimension is too high, so preprocessing is required. I compressed the original "neuron" dimension information into two features, mean and standard deviation (sd), which were previously proven to be related to success rate and time, thereby reducing the dimension of the data.

### 1. Original data structure
Each session has a spks list, where each element (trial) is a matrix with a dimension of (#neurons, 40) (assuming that 0.4 seconds is divided into 40 time bins).
brain_area corresponds to neurons, and its length is equal to the number of neurons. It is used to mark the brain area to which each neuron belongs.

### 2. Processing of a single trial (get_trial_features function)
For each trial, the function extracts the spike matrix of the corresponding neuron by brain area (brain_area) (size: number of neurons ?? 40 time steps), calculates the mean and standard deviation of each time step, and merges them into a 2??40 matrix.

### 3. Merge all brain regions into (num_all_brain, 40, 2)
Use get_trial_features to generate an array for each trial, the size of the array is (num_all_brain, 40, 2):
where num_all_brain is the number of brain regions that appear in all sessions;
40 represents the number of time steps;
2 represents two statistical features (mean and standard deviation) for each time step.

### 4 Integrate all trials
Finally, use abind(..., along=0) to merge the three-dimensional tensors of all trials into a four-dimensional tensor (n_trials, num_all_brain, 40, 2). Where n_trials is the total number of trials after all sessions are merged.

### 5 Data separation
The data is divided into 80% training and 20% validation.
```{r}


session <- list()
for(i in 1:18){
  session[[i]] <- readRDS(paste0('./Data/session', i, '.rds'))
}

get_trial_features <- function(s, trial_id) {
  trial_spks <- s$spks[[trial_id]]    
  brain_areas <- s$brain_area        
  unique_brains <- unique(brain_areas)  
  
  features_list <- list()
  for(area in unique_brains) {
    idx <- which(brain_areas == area)
    sub_mat <- trial_spks[idx, , drop = FALSE] 
    mean_vec <- colMeans(sub_mat)
    sd_vec <- apply(sub_mat, 2, sd)
    features_list[[area]] <- rbind(mean_vec, sd_vec)
  }
  return(features_list)
}

all_brain_areas <- unique(unlist(lapply(session, function(s) s$brain_area)))
num_all_brain <- length(all_brain_areas)  
time_steps <- 40  
features <- 2    

trial_data_list <- list()   
trial_labels <- c()         
trial_aux <- list()         

for(s in session) {
  n_trials <- length(s$spks)
  for(i in 1:n_trials) {
    trial_features <- get_trial_features(s, i)
    trial_array <- array(0, dim = c(num_all_brain, time_steps, features))
    for(j in 1:num_all_brain) {
      area <- all_brain_areas[j]
      if(area %in% names(trial_features)) {
        trial_array[j, , ] <- trial_features[[area]]
      }
    }
    trial_data_list[[length(trial_data_list) + 1]] <- trial_array
    label <- ifelse(s$feedback_type[i] == 1, 1, 0)
    trial_labels <- c(trial_labels, label)
    trial_aux[[length(trial_aux) + 1]] <- c(s$contrast_left[i], s$contrast_right[i])
  }
}

brain_data_array <- abind::abind(trial_data_list, along = 0)
aux_data <- do.call(rbind, trial_aux)
trial_labels <- as.numeric(trial_labels)

set.seed(123)
n_samples <- dim(brain_data_array)[1]
indices <- sample(1:n_samples)
train_idx <- indices[1:round(0.8 * n_samples)]
val_idx <- indices[(round(0.8 * n_samples) + 1):n_samples]

brain_train <- brain_data_array[train_idx, , , ]
aux_train <- aux_data[train_idx, ]
labels_train <- trial_labels[train_idx]

brain_val <- brain_data_array[val_idx, , , ]
aux_val <- aux_data[val_idx, ]
labels_val <- trial_labels[val_idx]
```

## Build the CNN model
In my CNN model, I considered the temporal features in the neuronal spike data (the average spike rate and standard deviation of each brain region over 40 time steps were extracted through 1D convolution) and visual stimulus information (left-right contrast). By extracting the local temporal patterns of the data of each brain region and integrating the global features of all brain regions, the model can learn the relationship between the activities of different brain regions and the experimental results, so as to predict the feedback type. This way of processing multi-dimensional temporal data gives the model an advantage in capturing the potential association between complex neural activity patterns and behavioral decisions. Finally, after two fully connected layers and a Sigmoid output layer, the model outputs a binary classification result (success or failure) to predict the feedback type of the experiment.

### Input layer design:
Brain region data input:
Define an input layer input_layer with a shape of (num_all_brain, 40, 2), which represents the 2 features of each brain region at 40 time steps in each trial.
Auxiliary input layer:
Define an auxiliary input layer aux_input with a shape of (2), which is used to input left and right contrast information.

### CNN branch design:
Define a separate CNN branch (cnn_branch), treating the data of each brain region as 1D time series data:
First layer: 1D convolution layer (16 filters, convolution kernel size of 3, ReLU activation, same padding);
Then perform batch normalization and maximum pooling (pooling size of 2);
Second layer: add another 1D convolution layer (32 filters), and perform batch normalization as well;
Finally, use global average pooling (Global Average Pooling 1D) to summarize the time series features of each brain region into a vector of fixed length (32 dimensions).

### TimeDistributed layer:
The time_distributed layer is used to apply the CNN branch defined above to each brain region, so as to extract features from the time series data of all brain regions in each trial, and the output shape is (n_trials, num_all_brain, 32).

### Global aggregation layer:
The output feature vectors of all brain regions are globally averaged (layer_global_average_pooling_1d) to integrate the features of different brain regions into a fixed-length feature vector.

### Auxiliary input and feature concatenation:
The globally summarized brain region features and auxiliary input (left and right contrast) are concatenated through layer_concatenate to form a richer feature representation.

### Fully connected layer (FC):
The concatenated feature vector passes through two fully connected layers:
The first layer: 64 neurons, ReLU activation, and Dropout is used to prevent overfitting;
The second layer: 32 neurons, ReLU activation.

### Output layer:
Finally, it passes through a Dense layer and uses Sigmoid activation to output a value between 0 and 1 for binary classification (success and failure).



```{r}

input_layer <- layer_input(shape = c(num_all_brain, time_steps, features), 
                           name = "brain_area_input")


cnn_branch <- keras_model_sequential(name = "cnn_branch") %>%
  layer_conv_1d(filters = 16, kernel_size = 3, activation = "relu", padding = "same",
                input_shape = c(time_steps, features)) %>%
  layer_batch_normalization() %>%  
  layer_max_pooling_1d(pool_size = 2) %>%  
  layer_conv_1d(filters = 32, kernel_size = 3, activation = "relu", padding = "same") %>%
  layer_batch_normalization() %>%
  layer_global_average_pooling_1d()

```

## Model layers and training
```{r}

if (!requireNamespace("rstudioapi", quietly = TRUE)) {
  install.packages("rstudioapi")
}
library(rstudioapi)
if ("readRStudioPreference" %in% ls(asNamespace("rstudioapi"))) {
  unlockBinding("readRStudioPreference", asNamespace("rstudioapi"))
  assign("readRStudioPreference", function(name, default) { default },
         envir = asNamespace("rstudioapi"))
  lockBinding("readRStudioPreference", asNamespace("rstudioapi"))
}

if ("package:tfruns" %in% search()){
  detach("package:tfruns", unload = TRUE)
}



td_layer <- input_layer %>% 
  time_distributed(cnn_branch)

aggregated_features <- td_layer %>% 
  layer_global_average_pooling_1d()

aux_input <- layer_input(shape = c(2), name = "aux_input")
merged_features <- layer_concatenate(list(aggregated_features, aux_input))

fc <- merged_features %>%
  layer_dense(units = 64, activation = "relu") %>%  
  layer_dropout(rate = 0.5) %>%                    
  layer_dense(units = 32, activation = "relu")    

output_layer <- fc %>% 
  layer_dense(units = 1, activation = "sigmoid")

model <- keras_model(
  inputs = list(input_layer, aux_input),
  outputs = output_layer
)

model %>% compile(
  optimizer = optimizer_adam(learning_rate = 0.001),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

summary(model)


history <- model %>% fit(
  x = list(brain_area_input = brain_train, aux_input = aux_train),
  y = labels_train,
  epochs = 10,             
  batch_size = 32,        
  validation_data = list(list(brain_area_input = brain_val, aux_input = aux_val), labels_val)
)


```




test the model's performance based of the test data
Test data 1
```{r}

test_session1 <- readRDS("C:/Users/howar/iCloudDrive/STA 141A/Project/test/test1.rds")

test_trial_data_list1 <- list()  
test_trial_labels1 <- c()         
test_trial_aux1 <- list()        

n_trials1 <- length(test_session1$spks)
for(i in 1:n_trials1) {
  trial_features <- get_trial_features(test_session1, i)
  trial_array <- array(0, dim = c(num_all_brain, time_steps, features))
  for(j in 1:num_all_brain) {
    area <- all_brain_areas[j]
    if(area %in% names(trial_features)) {
      trial_array[j, , ] <- trial_features[[area]]
    }
  }
  test_trial_data_list1[[length(test_trial_data_list1) + 1]] <- trial_array
  label <- ifelse(test_session1$feedback_type[i] == 1, 1, 0)
  test_trial_labels1 <- c(test_trial_labels1, label)
  test_trial_aux1[[length(test_trial_aux1) + 1]] <- c(test_session1$contrast_left[i],
                                                      test_session1$contrast_right[i])
}

test_brain_data1 <- abind::abind(test_trial_data_list1, along = 0)
test_aux_data1 <- do.call(rbind, test_trial_aux1)
test_labels1 <- as.numeric(test_trial_labels1)

cat("Number of test1 data samples:", dim(test_brain_data1)[1], "\n")
cat("Number of brain regions per sample:", dim(test_brain_data1)[2], "\n")
cat("Time Bine: ", dim(test_brain_data1)[3], "\n")
cat("Number of features:", dim(test_brain_data1)[4], "\n")

predictions1 <- model %>% predict(list(brain_area_input = test_brain_data1,
                                         aux_input = test_aux_data1))
predicted_labels1 <- ifelse(predictions1 > 0.5, 1, 0)


eval_metrics1 <- model %>% evaluate(
  x = list(brain_area_input = test_brain_data1, aux_input = test_aux_data1),
  y = test_labels1
)
cat("test1.rds Evaluation results:\n")
print(eval_metrics1)

```
Test data 2
```{r}

test_session2 <- readRDS("C:/Users/howar/iCloudDrive/STA 141A/Project/test/test2.rds")

test_trial_data_list2 <- list()   
test_trial_labels2 <- c()         
test_trial_aux2 <- list()         

n_trials2 <- length(test_session2$spks)
for(i in 1:n_trials2) {
  trial_features <- get_trial_features(test_session2, i)

  trial_array <- array(0, dim = c(num_all_brain, time_steps, features))
  for(j in 1:num_all_brain) {
    area <- all_brain_areas[j]
    if(area %in% names(trial_features)) {
      trial_array[j, , ] <- trial_features[[area]]
    }
  }
  test_trial_data_list2[[length(test_trial_data_list2) + 1]] <- trial_array
  label <- ifelse(test_session2$feedback_type[i] == 1, 1, 0)
  test_trial_labels2 <- c(test_trial_labels2, label)
  test_trial_aux2[[length(test_trial_aux2) + 1]] <- c(test_session2$contrast_left[i],
                                                      test_session2$contrast_right[i])
}

test_brain_data2 <- abind::abind(test_trial_data_list2, along = 0)
test_aux_data2 <- do.call(rbind, test_trial_aux2)
test_labels2 <- as.numeric(test_trial_labels2)

cat("Number of test1 data samples:", dim(test_brain_data2)[1], "\n")
cat("Number of brain regions per sample:", dim(test_brain_data2)[2], "\n")
cat("Time Bine: ", dim(test_brain_data2)[3], "\n")
cat("Number of features:", dim(test_brain_data2)[4], "\n")

predictions2 <- model %>% predict(list(brain_area_input = test_brain_data2,
                                         aux_input = test_aux_data2))
predicted_labels2 <- ifelse(predictions2 > 0.5, 1, 0)


eval_metrics2 <- model %>% evaluate(
  x = list(brain_area_input = test_brain_data2, aux_input = test_aux_data2),
  y = test_labels2
)
cat("test2.rds Evaluation results:\n")
print(eval_metrics2)



```


# Reference
Steinmetz, N.A., Zatka-Haas, P., Carandini, M. et al. Distributed coding of choice, action and engagement across the mouse brain. Nature 576, 266?C273 (2019). https://doi.org/10.1038/s41586-019-1787-x

This project is talked with chatGPT with link: https://chatgpt.com/share/67d8b19d-c898-800c-ac53-567e4c190782

```{r}
rmarkdown::pandoc_version()
```








































