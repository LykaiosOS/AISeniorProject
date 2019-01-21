import numpy as np
import sys
import os
import tensorflow as tf
import pandas as pd
import math 
import IPython as display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
from sklearn import metrics
from tensorflow.python.data import Dataset

print("Booting Software May Take a Long Time")
###################################################################
# Variables                                                       #
# When launching project or scripts from Visual Studio,           #
# input_dir and output_dir are passed as arguments automatically. #
# Users could set them from the project setting page.             #
###################################################################

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("input_dir", ".", "Input directory where training dataset and meta data are saved")
tf.app.flags.DEFINE_string("output_dir", ".", "Output directory where output such as logs are saved.")
tf.app.flags.DEFINE_string("log_dir", ".", "Model directory where final model files are saved.")
print("Before Main")
# Features to examine.
cancerDataframe = pd.read_csv("data.csv")
print("Past CSV")
    # Randomizing Data.

cancerDataframe.reindex(np.random.permutation(cancerDataframe.index))
print("Past Reindex")
    # Examining Data.

cancerDataframe.describe()
print("Past Describe")            
selectedFeatures = cancerDataframe[
            ["diagnosis",
            "radius_mean",
            "texture_mean",
            "perimeter_mean",
            "area_mean",
            "smoothness_mean",
            "compactness_mean",
            "concavity_mean",
            "concave points_mean",
            "symmetry_mean",
            "fractal_dimension_mean"
            ]]
trainingExamples = selectedFeatures
validationExamples = cancerDataframe[["diagnosis",
            "radius_mean",
            "texture_mean",
            "perimeter_mean",
            "area_mean",
            "smoothness_mean",
            "compactness_mean",
            "concavity_mean",
            "concave points_mean",
            "symmetry_mean",
            "fractal_dimension_mean"]]

def main(_):
    print("Hi")
    # Adding code here.
    with tf.Session() as sess:
        loading = sess.run(tf.constant("Booting Tensorflow, please wait it may take a long time."))
    print(loading)
    # Add boot code here. 
    # Adding Dataset.

# Create Synthetic Features here.
    preprocessTargets(cancerDataframe, selectedFeatures, validationExamples, trainingExamples)
    userInputDataframe = userInput(cancerDataframeUI = [])
    checkingAnswers(userInputDataframe)
    learningRate=0.000003
    steps=20000
    batchSize=500
    trainingTargets = selectedFeatures
    validationTargets = selectedFeatures
    my_feature = constructFeatureColumns(selectedFeatures)
    linearRegressor, trainingInputFunctionRadius, predictTrainingInputRadius, predictValidationInputFunctionRadius, stepsPerPeriod = trainModel(learningRate,
    steps,
    batchSize,
    trainingExamples,
    trainingTargets,
    validationExamples,
    validationTargets,
    my_feature)
    recursive(linearRegressor, trainingInputFunctionRadius, predictTrainingInputRadius, predictValidationInputFunctionRadius, stepsPerPeriod)
    defined()
            
def preprocessTargets(cancerDataframe, selectedFeatures, validationExamples, trainingExamples):
        print("Targets")
    # Prepares Target Features.

        #outputTargets = pd.DataFrame.from_csv("data.csv")
        outputTargets = {key:np.array(value) for key,value in dict(selectedFeatures).items()}
        validationExamplesDict = {key:np.array(value) for key,value in dict(validationExamples).items()}
        validationTargets = {key:np.array(value) for key,value in dict(validationExamples).items()}
        trainingTargets = {key:np.array(value) for key,value in dict(trainingExamples).items()}
    # Scale stuff here.

        outputTargets["diagnosis"] = (
        cancerDataframe["diagnosis"])
        print(outputTargets)
        # Plotting Graph of Radius and Area.

        plt.figure(figsize=(21, 100))

        ax = plt.subplot(1, 2, 1)

        ax.set_title("Validation Data Radius/Area")
        ax.set_autoscaley_on = False
        ax.set_ylim(0, 250)
        ax.set_autoscalex_on = False
        ax.set_xlim(2, 13)
        plt.scatter(validationExamplesDict["radius_mean"],
        validationExamplesDict["area_mean"],
        cmap = "coolwarm",
        )

        ax = plt.subplot(1,2,2)
        ax.set_title("Training Data Radius/Area")

        ax.set_autoscaley_on(False)
        ax.set_ylim([251, 569])
        ax.set_autoscalex_on(False)
        ax.set_xlim([2, 13])
        plt.scatter(trainingTargets["radius_mean"],
        trainingTargets["area_mean"],
        cmap="coolwarm",
        )
        _ = plt.plot()

        plt.figure(figsize=(21, 100))

        bx = plt.subplot(1, 2, 1)

        bx.set_title("Validation Data Texture/Perimeter")
        bx.set_autoscaley_on = False
        bx.set_ylim(0, 250)
        bx.set_autoscalex_on = False
        bx.set_xlim(2, 13)
        plt.scatter(validationExamplesDict["texture_mean"],
                    validationExamplesDict["perimeter_mean"],
                    cmap = "coolwarm",
                    
                    )

        bx = plt.subplot(1,2,2)
        bx.set_title("Training Data Texture/Perimeter")

        bx.set_autoscaley_on(False)
        bx.set_ylim([251, 569])
        bx.set_autoscalex_on(False)
        bx.set_xlim([2, 13])
        plt.scatter(trainingTargets["texture_mean"],
            trainingTargets["perimeter_mean"],
            cmap="coolwarm",
            )
        _ = plt.plot()

        plt.figure(figsize=(21, 100))

        cx = plt.subplot(1, 2, 1)

        cx.set_title("Validation Data Smoothness/Compactness")
        cx.set_autoscaley_on = False
        cx.set_ylim(0, 250)
        cx.set_autoscalex_on = False
        cx.set_xlim(2, 13)
        plt.scatter(validationExamplesDict["smoothness_mean"],
                    validationExamplesDict["compactness_mean"],
                    cmap = "coolwarm",
                    
                    )

        cx = plt.subplot(1,2,2)
        cx.set_title("Training Data Smoothness/Compactness")

        cx.set_autoscaley_on(False)
        cx.set_ylim([251, 569])
        cx.set_autoscalex_on(False)
        cx.set_xlim([2, 13])
        plt.scatter(trainingTargets["smoothness_mean"],
            trainingTargets["compactness_mean"],
            cmap="coolwarm",
            )
        _ = plt.plot()

        plt.figure(figsize=(21, 100))

        cx = plt.subplot(1, 2, 1)

        cx.set_title("Validation Data Concavity/Concave Points")
        cx.set_autoscaley_on = False
        cx.set_ylim(0, 250)
        cx.set_autoscalex_on = False
        cx.set_xlim(2, 13)
        plt.scatter(validationExamplesDict["concavity_mean"],
                    validationExamplesDict["concave points_mean"],
                    cmap = "coolwarm",
                    
                    )

        cx = plt.subplot(1,2,2)
        cx.set_title("Training Data Concavity/Concave Points")

        cx.set_autoscaley_on(False)
        cx.set_ylim([251, 569])
        cx.set_autoscalex_on(False)
        cx.set_xlim([2, 13])
        plt.scatter(trainingTargets["concavity_mean"],
            trainingTargets["concave points_mean"],
            cmap="coolwarm",
            )
        _ = plt.plot()
        # Done Plotting.
        
        welcome = "Booted Successfully"
        print(welcome)

        # Main Code goes here.
        
        print("Past Plots")

def cancerTrainingModel(features, targets, batchSize = [1,500], shuffle = True, numEpochs = None):
            # Training the function with multiple variables.

            # Converting the pandas library into a dict of np arrays.

            features = construct_feature_columns(selectedFeatures)
            targets = construct_target_columns(selectedFeatures)
            features = {key:np.array(value) for key,value in dict(features).items()}
            #targets = {key:np.array(value) for key,value in dict(features).items()}
            #batch = {'batch1':[1, 500], 'batch2':[501, 1000] }
            # Constucting a dataset.

            #ds = pd.DataFrame(features,batch)
            construct_feature_columns()
            ds = Dataset.from_tensor_slices((features,targets))
            ds = ds.batch(batchSize).repeat(numEpochs)

            # Suffles data if true.
            if shuffle:
                ds = ds.shuffle(1000)

            # Returns the next batch.

            features, labels = ds.make_one_shot_iterator().get_next()
            return features, labels

def constructFeatureColumns(selectedFeatures):

  """Construct the TensorFlow Feature Columns.

  Args:
    input_features: The names of the numerical input features to use.
  Returns:
    A set of feature columns
  """ 
  return set([tf.feature_column.numeric_column(my_feature)
              for my_feature in selectedFeatures])

def construct_target_columns(input_targets):

  """Construct the TensorFlow Feature Columns.

  Args:
    input_features: The names of the numerical input features to use.
  Returns:
    A set of feature columns
  """ 
  return set([tf.feature_column.numeric_column(my_target)
              for my_target in input_targets])

def trainModel(
    learningRate,
    steps,
    batchSize,
    trainingExamples,
    trainingTargets,
    validationExamples,
    validationTargets,
    my_feature):

    steps = 3
    periods = 10
    stepsPerPeriod = steps / periods

    # Creates a linear regression object.

    cancerOptimizer = tf.train.GradientDescentOptimizer(learningRate)
    cancerOptimizer = tf.contrib.estimator.clip_gradients_by_norm(cancerOptimizer, 5.0)
    linearRegressor = tf.estimator.LinearRegressor(
        feature_columns = "diagnosis",
        optimizer = cancerDataframe
    )

    # Creating Input Functions.

    trainingInputFunctionRadius = lambda: diagnosisInputFunction(
        trainingExamples,
        trainingTargets["radius_mean"],
        batchSize = batchSize)
    predictTrainingInputRadius = lambda: diagnosisInputFunction(
        trainingExamples,
        trainingTargets["radius_mean"],
        numEpochs = 1,
        shuffle = False)
    predictValidationInputFunctionRadius = lambda: diagnosisInputFunction(
        validationExamples, validationTargets["radius_Mean"],
        numEpochs = 1,
        shuffle = False)

    trainingInputFunctionPerimeter = lambda: diagnosisInputFunction(
        trainingExamples,
        trainingTargets["perimeter_mean"],
        batchSize = batchSize)
    predictTrainingInputPerimeter = lambda: diagnosisInputFunction(
        trainingExamples,
        trainingTargets["perimeter_mean"],
        numEpochs = 1,
        shuffle = False)
    predictValidationInputFunctionPerimeter = lambda: diagnosisInputFunction(
        validationExamples, validationTargets["perimeter_Mean"],
        numEpochs = 1,
        shuffle = False)

    trainingInputFunctionDiagnosisArea = lambda: diagnosisInputFunction(
        trainingExamples,
        trainingTargets["area_mean"],
        batchSize = batchSize)
    predictTrainingInputArea = lambda: diagnosisInputFunction(
        trainingExamples,
        trainingTargets["area_mean"],
        numEpochs = 1,
        shuffle = False)
    predictValidationInputFunctionArea = lambda: diagnosisInputFunction(
        validationExamples, validationTargets["area_Mean"],
        numEpochs = 1,
        shuffle = False)

    trainingInputFunctionSmoothness = lambda: diagnosisInputFunction(
        trainingExamples,
        trainingTargets["smoothness_mean"],
        batchSize = batchSize)
    predictTrainingInputSmoothness = lambda: diagnosisInputFunction(
        trainingExamples,
        trainingTargets["smoothness_mean"],
        numEpochs = 1,
        shuffle = False)
    predictValidationInputFunctionSmoothness = lambda: diagnosisInputFunction(
        validationExamples, validationTargets["smoothness_Mean"],
        numEpochs = 1,
        shuffle = False)

    trainingInputFunctionCompactness = lambda: diagnosisInputFunction(
        trainingExamples,
        trainingTargets["compactness_mean"],
        batchSize = batchSize)
    predictTrainingInputCompactness = lambda: diagnosisInputFunction(
        trainingExamples,
        trainingTargets["compactness_mean"],
        numEpochs = 1,
        shuffle = False)
    predictValidationInputFunctionCompactness = lambda: diagnosisInputFunction(
        validationExamples, validationTargets["compactness_Mean"],
        numEpochs = 1,
        shuffle = False)  
    return linearRegressor, trainingInputFunctionRadius, predictTrainingInputRadius, predictValidationInputFunctionRadius, stepsPerPeriod

def defined():
            print("Training Model with new user inputs.")
            print("RMSE on the training data")
            trainingRMSE = []
            validationRMSE = []
            for period in range (0, periods):
                linearRegressor.train(
                inputFunction = trainingInputFunction,
                steps = stepsPerPeriod,
                )
            trainingPredictions = linearRegressor.predict(inputFunction = predictTrainingInputFunction)
            trainingPredictions = np.array([item['predictions'][0] for item in trainingPredictions])

            validationPredictions = linearRegressor.predict(inputFunction = predictValidationInputFunction)
            validationPredictions = np.array([item['predictions'][0] for item in validationPredictions])

            # Compute taining and validation loss.

            trainingRootMeanSquarredError = math.sqrt(
                metrics.meanSquarredError(trainingPredictions, trainingTargets)
                )
            validationRootMeanSquarredError = math.sqrt(
                metrics.meanSquarredError(validationPredictions, validationTargets)
                )
            test_examples = preprocess_features(cancerDataframe)
            test_targets = preprocess_targets(cancerDataframe)

            predict_test_input_fn = lambda: cancerInput(
            test_examples, 
            test_targets["compactness_mean"], 
            num_epochs=1, 
            shuffle=False)

            test_predictions = linear_regressor.predict(input_fn=predict_test_input_fn)
            test_predictions = np.array([item['predictions'][0] for item in test_predictions])

            rootMeanSquaredErrorNew = math.sqrt(
            metrics.mean_squared_error(test_predictions, test_targets))


            print("Final RMSE (on test data): %0.2f" % rootMeanSquaredErrorNew)

            linear_classifier = train_linear_classifier_model(
            learning_rate=0.000003,
            steps=20000,
            batch_size=500,
            training_examples=training_examples,
            training_targets=training_targets,
            validation_examples=validation_examples,
            validation_targets=validation_targets)

            evaluation_metrics = linear_classifier.evaluate(input_fn=predict_validation_input_fn)

            print("AUC on the validation set: %0.2f" % evaluation_metrics['auc'])
            print("Accuracy on the validation set: %0.2f" % evaluation_metrics['accuracy'])
            def select_and_transform_features(source_df):
                DIAGNOSIS = zip(range(32, 44), range(33, 45))
                selected_examples = pd.DataFrame()
                selected_examples["diagnosis"] = source_df["diagnosis"]
                for r in DIAGNOSIS:
                    selected_examples["Diagnosis_%d_to_%d" % r] = source_df["diagnosis"].apply(
                    lambda l: 1.0 if l >= r[0] and l < r[1] else 0.0)
                return selected_examples

                selected_training_examples = select_and_transform_features(training_examples)
                selected_validation_examples = select_and_transform_features(validation_examples)
                 
            recursive = [...]  # Populate with the output of the last run
            defined = [...]  # The current results

            # Remove rows that aren't in the current result set
            for row in recursive - defined:
                deleteentry(row[0])  # Where row[0] is the unique key for the table

            # Add rows that weren't in the last result set
            for row in recursive - defined:
                insertentry(row)

            if defined != recursive:
                print("Chances are the node is malignant.")
            else:
                print("Chances are the node is benign.")

def recursive(linearRegressor, trainingInputFunctionRadius, predictTrainingInputRadius, predictValidationInputFunctionRadius, stepsPerPeriod):

            # Choose the first 200 (out of 576) examples for training.
            training_examples = cancerDataframe.head(200)
            training_targets = cancerDataframe.head(200)

            # Choose the last 376 (out of 576) examples for validation.
            validation_examples = cancerDataframe.tail(376)
            validation_targets = cancerDataframe.tail(376)

            # Double-check that it has done the right thing.
            

            print("Training Model")
            print("RMSE on the training data")
            trainingRMSE = []
            validationRMSE = []
            inputFunction = []
            periods = 10
            for period in range (0, periods):
                linearRegressor.train(
                    inputFunction = trainingInputFunctionRadius,
                    steps = stepsPerPeriod
                    )
                trainingPredictions = linearRegressor.predict(inputFunction = predictTrainingInputFunctionRadius)
                trainingPredictions = np.array([item['predictions'][0] for item in trainingPredictions])

                validationPredictions = linearRegressor.predict(inputFunction = predictValidationInputFunctionRadius)
                validationPredictions = np.array([item['predictions'][0] for item in validationPredictions])

                # Compute taining and validation loss.

                trainingRootMeanSquarredError = math.sqrt(
                    metrics.meanSquarredError(trainingPredictions, trainingTargets)
                    )
                validationRootMeanSquarredError = math.sqrt(
                    metrics.meanSquarredError(validationPredictions, validationTargets)
                    )
                test_examples = preprocess_features(cancerDataframe)
                test_targets = preprocess_targets(cancerDataframe)

                predict_test_input_fn = lambda: cancerInput(
                test_examples, 
                test_targets["daignosis"], 
                num_epochs=1, 
                shuffle=False)

                test_predictions = linear_regressor.predict(input_fn=predict_test_input_fn)
                test_predictions = np.array([item['predictions'][0] for item in test_predictions])

                rootMeanSquaredError = math.sqrt(
                    metrics.mean_squared_error(test_predictions, test_targets))

                print("Final RMSE (on test data): %0.2f" % rootMeanSquaredError)

def userInput(cancerDataframeUI = []):

            # User inputs get inputted here.

            radiusMeanUI = input("Please input the radius mean: ")
            textureMeanUI = input("Please input the texture mean: ")
            perimeterMeanUI = input("Please input perimeter mean: ")
            areaMeanUI = input("Please input area mean: ")
            smoothnessMeanUI = input("Please input smoothness mean: ")
            compactnessMeanUI = input("Please input compactness mean: ")
            concavityMeanUI = input("Please input concavity mean: ")
            concavePointsMeanUI = input("Please input concave points mean: ")
            concaveMeanUI = input("Please input concave mean: ")
            symmetryMeanUI = input("Please input symmetry mean: ")
            fractalDimensionUI = input("Please input fractal dimension mean: ")

            # Setting up the user input array.

            return [radiusMeanUI, textureMeanUI, perimeterMeanUI, areaMeanUI, smoothnessMeanUI, compactnessMeanUI, concavityMeanUI, concaveMeanUI, symmetryMeanUI, fractalDimensionUI]
            

def checkingAnswers(userInputDataframe):
            
    #Checking Answers.

    print("Information inputted into the system. Make sure all of the information is correct.")
    print("1: Radius Mean: " + userInputDataframe[0])
    print("2: Texture Mean: " + userInputDataframe[1])
    print("3: Perimeter Mean: " + userInputDataframe[2])
    print("4: Area Mean: " + userInputDataframe[3])
    print("5: Smoothness Mean: " + userInputDataframe[4])
    print("6: Compactness Mean: " + userInputDataframe[5])
    print("7: Concavity Mean: " + userInputDataframe[6])
    print("8: Concave Points Mean: " + userInputDataframe[7])
    print("9: Symmetry Mean: " + userInputDataframe[8])
    print("10 Fractal Mean: " + userInputDataframe[9])

    # Asks user to point out errors.

    print("Are all of these correct? Type Yes or No. If one is not correct please state which one is incorrect by typing " r"1" " for example.")

    wrongUserInputCheck = input("Yes/No: ")
    if wrongUserInputCheck == "Yes":
            
        if wrongUserInputCheck == "No":
            wrongUserInputAnswer = input("Which input is wrong: ")
            if wrongUserInputAnswer == wrongUserInputAnswer:
                wrongNewUserInput = input("Please input correct " + wrongUserInputAnswer + ":")
                print("Does this look correct? " + wrongUserNewInput)
                checkAgainUserInput = input("Yes/No: ")
                    
                if checkAgainUserInput == "Yes":
                    print("Adding Newest Input.")
                    wrongUserInputAnswer - 1
                    userInputDataframe[wrongUserInputAnswer] = wrongUserNewInput
                    print("Was there any other error that occured?")
                    checkToSeeIfOtherErrror = input("Yes/No: ")

                    if checkToSeeIfOtherError == "Yes":
                        checkingAnswers() 
                        
                    if checkToSeeIfOtherErrors == "No":
                        return checkingAnswers(0)

                if checkAgainUserInput == "No":
                    userNewInput = input("Adding new values to arrray.")
                    ax + userInputDataframe[0],userInputDataframe[3]
                    bx + userInputDataframe[4], userInputDataframe[5]
                    cx + userInputDataframe[6], userInputDataframe[7]
print("After Main")
if __name__ == '__main__':
   tf.app.run()
