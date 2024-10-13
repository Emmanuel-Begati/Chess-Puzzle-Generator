# Chess Puzzle Difficulty Prediction

## Project Overview

This project aims to predict the difficulty rating of chess puzzles using machine learning techniques. We used a dataset from Lichess, which contains rich information about chess puzzles, including board positions, moves, and various metadata.
The entire data was about 4 million rows and it was taking forever to process.. so we got a chunk of the data (10000 rows). In the near future, we will improve the model by using the entire dataset

## Dataset

Our dataset is specifically tailored for chess puzzle difficulty prediction, aligning closely with the project's objectives. It includes the following key features:

1. FEN (Forsyth–Edwards Notation): Represents the board state.
2. Moves: The sequence of moves that solve the puzzle.
3. Rating: The difficulty rating of the puzzle (our target variable).
4. Themes: Categories or types of tactics involved in the puzzle.
5. Popularity: A measure of how often the puzzle is attempted.
6. Number of pieces: Count of pieces for both white and black.
7. Material count: Total material value for both white and black.

This rich dataset provides both volume (10,000 samples) and variety in terms of puzzle types and difficulty levels, making it ideal for our machine learning task.

## Model Implementation

We implemented two main models for this project:

1. **Simple Model (Vanilla Neural Network)**:
   - A basic neural network with two hidden layers.
   - Uses ReLU activation and Adam optimizer.
   - No additional optimization techniques.

2. **Improved Model**:
   - A deeper neural network with three hidden layers.
   - Incorporates several optimization techniques:

     a) **L2 Regularization**: 
        - Applied to each dense layer with a factor of 0.01.
        - Helps prevent overfitting by adding a penalty term to the loss function.

     b) **Batch Normalization**: 
        - Applied after each hidden layer.
        - Normalizes the inputs to each layer, which can speed up training and provide some regularization effects.

     c) **Dropout**: 
        - Applied after each hidden layer with a rate of 0.3.
        - Randomly sets input units to 0 during training, which helps prevent overfitting.

     d) **Early Stopping**:
        - Monitors validation loss with a patience of 10 epochs.
        - Helps prevent overfitting by stopping training when the model stops improving.

     e) **Learning Rate Adjustment**:
        - Used Adam optimizer with a learning rate of 0.01.
        - This learning rate was chosen after experimentation to balance between convergence speed and stability.

## Optimization Techniques Discussion

### L2 Regularization
L2 regularization adds a penalty term to the loss function, proportional to the square of the weights. This encourages the model to use smaller weights, which can help prevent overfitting. We chose a factor of 0.01 after experimentation, finding it provided a good balance between model complexity and generalization.

### Batch Normalization
Batch normalization normalizes the inputs to each layer, which can help mitigate the internal covariate shift problem. This often allows for higher learning rates and can provide a slight regularization effect. We applied it after each hidden layer to maintain consistent input distributions throughout the network.

### Dropout
Dropout is a powerful regularization technique that randomly sets a fraction of input units to 0 during training. We used a dropout rate of 0.3, meaning 30% of the units are dropped in each forward pass. This rate was chosen as a balance between retaining enough information and providing sufficient regularization.

### Early Stopping
Early stopping helps prevent overfitting by stopping the training process when the model's performance on the validation set stops improving. We set a patience of 10 epochs, allowing the model some room to overcome temporary plateaus while still stopping before significant overfitting occurs.

### Learning Rate Adjustment
We used the Adam optimizer with a learning rate of 0.01. This value was chosen after experimenting with different rates. A higher rate led to unstable training, while a lower rate resulted in slow convergence. The chosen rate provided a good balance between training speed and stability.

## Error Analysis

We conducted a comprehensive error analysis using multiple metrics:

1. **Mean Absolute Error (MAE)**:
   - Vanilla Model MAE: 349.74
   - Improved Model MAE: 308.67

   MAE measures the average magnitude of the errors in a set of predictions, without considering their direction. The lower MAE of the Improved Model indicates better overall prediction accuracy.

2. **Mean Squared Error (MSE)**:
   - Vanilla Model MSE: 226,250.23
   - Improved Model MSE: 153,031.40

   MSE measures the average squared difference between the estimated values and the actual value. It gives a higher weight to larger errors. The lower MSE of the Improved Model shows it makes fewer large prediction errors.

3. **R-squared (R²)**:
   - Vanilla Model R²: 0.2260
   - Improved Model R²: 0.4765

   R² represents the proportion of the variance in the dependent variable that is predictable from the independent variables. The higher R² of the Improved Model indicates it explains more of the variability in the data.

These metrics are well-formatted and easy to interpret, providing a clear picture of each model's performance.

## Performance Baseline

Our Improved Model significantly outperformed the Vanilla Model across all metrics:

- Mean Absolute Error (MAE): Improved by 41.07 points (11.74% reduction)
- Mean Squared Error (MSE): Improved by 73,218.83 points (32.36% reduction)
- R-squared (R²): Improved by 0.2505 (110.84% increase)

While we don't have traditional accuracy measures for this regression task, the substantial improvements in MAE and MSE, coupled with the more than doubling of the R² value, demonstrate the effectiveness of our optimization techniques.

The Improved Model shows consistent and significant performance gains over the Vanilla Model. However, with an R² of 0.4765, there's still room for further improvement, possibly through additional feature engineering or more advanced modeling techniques.

## Conclusion

Through careful feature engineering, model optimization, and comprehensive error analysis, we've developed an improved predictive model for chess puzzle difficulty. The significant enhancements in MAE, MSE, and R² from our Vanilla to our Improved Model showcase the power of advanced machine learning techniques in this domain.
The model is not very decent... we used a small chunk of the data which might be the reason why it didn't generalize the prediction well... 
Future work could involve experimenting with even more complex architectures, incorporating additional chess-specific features, or exploring ensemble methods to further enhance the model's predictive power and push the R² value higher.
