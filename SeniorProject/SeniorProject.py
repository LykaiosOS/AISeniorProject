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

def main(_):
    # Adding code here.
    with tf.Session() as sess:
        loading = sess.run(tf.constant("Booting Tensorflow, please wait it may take a long time."))
        print(loading)
        # Add boot code here. 

        # Adding Dataset.

        cancerDataframe = pd.read_csv("")

        # Randomizing Data.

        cancerDataframe.reindex(np.random.permutation(cancerDataset.index))

        # Examining Data.

        cancerDataframe.describe()
        def preprocessFetures(cancerDataframe):

            # Features to examine.

            selectedFeatures = cancerDataframe[
                ["diagnosis",
                "radius_mean",
                "texture_mean",
                "perimeter_mean",
                "area_mean",
                "smoothness_mean",
                "compactness_mean",
                "concavity_mean",
                "concave_points_mean",
                "symmetry_mean",
                "fractal_dimension_mean"
                  ]]
        processedFeatures = selectedFeatures.copy()
        return preprocessFetures

        # Create Synthetic Features here.

        def preprocessTargets(cancerDataframe):

            # Prepares Target Features.

            outputTargets = pd.Dataframe()

            # Scale stuff here.

            outputTargets["diagnosis"] = (
                cancerDataframe["diagnosis"])
            return outputTargets

        # Plotting Graph of Radius and Area.

        plt.figure(figsize=(21, 100))

        bx = plt.subplot(1, 2, 1)

        ax.set_title("Validation Data Radius/Area")
        ax.set_autoscaley_on = False
        ax.set_ylim(0, 250)
        ax.set_autoscalex_on = False
        ax.set_xlim(2, 13)
        plt.scatter(validationExamples["radius_mean"],
                    validationExamples["area_mean"],
                    cmap = "coolwarm",
                    c = validationTargets["diagnosis"] / validationTargets["diagnosis"].max()
                    )

        ax = plt.subplot(1,2,2)
        ax.set_title("Training Data Radius/Area")

        ax.set_autoscaley_on(False)
        ax.set_ylim([251, 569])
        ax.set_autoscalex_on(False)
        ax.set_xlim([2, 13])
        plt.scatter(training_examples["radius_mean"],
            training_examples["area_mean"],
            cmap="coolwarm",
            c=training_targets["diagnosis"] / training_targets["diagnosis"].max())
        _ = plt.plot()

        plt.figure(figsize=(21, 100))

        bx = plt.subplot(1, 2, 1)

        bx.set_title("Validation Data Texture/Perimeter")
        bx.set_autoscaley_on = False
        bx.set_ylim(0, 250)
        bx.set_autoscalex_on = False
        bx.set_xlim(2, 13)
        plt.scatter(validationExamples["texture_mean"],
                    validationExamples["perimeter_mean"],
                    cmap = "coolwarm",
                    c = validationTargets["diagnosis"] / validationTargets["diagnosis"].max()
                    )

        bx = plt.subplot(1,2,2)
        bx.set_title("Training Data Texture/Perimeter")

        bx.set_autoscaley_on(False)
        bx.set_ylim([251, 569])
        bx.set_autoscalex_on(False)
        bx.set_xlim([2, 13])
        plt.scatter(training_examples["texture_mean"],
            training_examples["perimeter_mean"],
            cmap="coolwarm",
            c=training_targets["diagnosis"] / training_targets["diagnosis"].max())
        _ = plt.plot()

        plt.figure(figsize=(21, 100))

        cx = plt.subplot(1, 2, 1)

        cx.set_title("Validation Data Smoothness/Compactness")
        cx.set_autoscaley_on = False
        cx.set_ylim(0, 250)
        cx.set_autoscalex_on = False
        cx.set_xlim(2, 13)
        plt.scatter(validationExamples["smoothness_mean"],
                    validationExamples["compactness_mean"],
                    cmap = "coolwarm",
                    c = validationTargets["diagnosis"] / validationTargets["diagnosis"].max()
                    )

        cx = plt.subplot(1,2,2)
        cx.set_title("Training Data Smoothness/Compactness")

        cx.set_autoscaley_on(False)
        cx.set_ylim([251, 569])
        cx.set_autoscalex_on(False)
        cx.set_xlim([2, 13])
        plt.scatter(training_examples["smoothness_mean"],
            training_examples["compactness_mean"],
            cmap="coolwarm",
            c=training_targets["diagnosis"] / training_targets["diagnosis"].max())
        _ = plt.plot()

        plt.figure(figsize=(21, 100))

        cx = plt.subplot(1, 2, 1)

        cx.set_title("Validation Data Concavity/Concave Points")
        cx.set_autoscaley_on = False
        cx.set_ylim(0, 250)
        cx.set_autoscalex_on = False
        cx.set_xlim(2, 13)
        plt.scatter(validationExamples["concavity_mean"],
                    validationExamples["concave_points_mean"],
                    cmap = "coolwarm",
                    c = validationTargets["diagnosis"] / validationTargets["diagnosis"].max()
                    )

        cx = plt.subplot(1,2,2)
        cx.set_title("Training Data Concavity/Concave Points")

        cx.set_autoscaley_on(False)
        cx.set_ylim([251, 569])
        cx.set_autoscalex_on(False)
        cx.set_xlim([2, 13])
        plt.scatter(training_examples["concavity_mean"],
            training_examples["concave_points_mean"],
            cmap="coolwarm",
            c=training_targets["diagnosis"] / training_targets["diagnosis"].max())
        _ = plt.plot()
        # Done Plotting.
        
        welcome = sess.run(tf.constant("Booted Successfuly."))
        print(welcome)

        # Main Code goes here.
        
        def cancerTrainingModel(features, batchSize = 1, shuffle = True, numEpochs = None):
            # Training the function with multiple variables.

            # Converting the pandas library into a dict of np arrays.

            features = {key:np.array(value) for key,value in dict(features).items()}

            # Constucting a dataset.

            ds = Dataset.from_tensor_slices((features,targets))
            ds = ds.batch(batchSize).repeat(numEpochs)

            # Suffles data if true.
            if shuffle:
                ds = ds.shuffle(1000)

            # Returns the next batch.

            features, labels = ds.make_one_shot_iterator().get_next()
            return features, labels

        def constructFeatureColumns(inputFeatures):
            
            return set([tf.feature_column.numeric.column(cancerFeatures)
                        for cancerFeatures in inputFeatures])

        def trainModel(
            learningRate,
            steps,
            batchSize,
            trainingExamples,
            trainingTargets,
            validationExamples,
            validationTargets):

            periods = 10
            stepsPerPeriod = steps / periods

            # Creates a linear regression object.

            cancerOptimizer = tf.train.GradientDescentOptimizer(learningRate = learningRate)
            cancerOptimizer = tf.contrib.estimator.clip_gradients_by_norm(cancerOptimizer, 5.0)
            linearRegressor = tf.estimator.LinearRegressor(
                featureColumns = constructFeatureColumns(trainingExamples),
                optimizer = cancerDataframe
            )

            # Creating Input Functions.

            trainingInputFunctionDiagnosis = lambda: diagnosisInputFunction(
                trainingExamples,
                trainingTargets["radius_mean"],
                batchSize = batchSize)
            predictTrainingInput = lambda: diagnosisInputFunction(
                trainingExamples,
                trainingTargets["radius_mean"],
                numEpochs = 1,
                shuffle = False)
            predictValidationInputFunction = lambda: diagnosisInputFunction(
                validationExamples, validationTargets["radius_Mean"],
                numEpochs = 1,
                shuffle = False)

            trainingInputFunctionDiagnosis = lambda: diagnosisInputFunction(
                trainingExamples,
                trainingTargets["perimeter_mean"],
                batchSize = batchSize)
            predictTrainingInput = lambda: diagnosisInputFunction(
                trainingExamples,
                trainingTargets["perimeter_mean"],
                numEpochs = 1,
                shuffle = False)
            predictValidationInputFunction = lambda: diagnosisInputFunction(
                validationExamples, validationTargets["perimeter_Mean"],
                numEpochs = 1,
                shuffle = False)

            trainingInputFunctionDiagnosis = lambda: diagnosisInputFunction(
                trainingExamples,
                trainingTargets["area_mean"],
                batchSize = batchSize)
            predictTrainingInput = lambda: diagnosisInputFunction(
                trainingExamples,
                trainingTargets["area_mean"],
                numEpochs = 1,
                shuffle = False)
            predictValidationInputFunction = lambda: diagnosisInputFunction(
                validationExamples, validationTargets["area_Mean"],
                numEpochs = 1,
                shuffle = False)

            trainingInputFunctionDiagnosis = lambda: diagnosisInputFunction(
                trainingExamples,
                trainingTargets["smoothness_mean"],
                batchSize = batchSize)
            predictTrainingInput = lambda: diagnosisInputFunction(
                trainingExamples,
                trainingTargets["smoothness_mean"],
                numEpochs = 1,
                shuffle = False)
            predictValidationInputFunction = lambda: diagnosisInputFunction(
                validationExamples, validationTargets["smoothness_Mean"],
                numEpochs = 1,
                shuffle = False)

            trainingInputFunctionDiagnosis = lambda: diagnosisInputFunction(
                trainingExamples,
                trainingTargets["compactness_mean"],
                batchSize = batchSize)
            predictTrainingInput = lambda: diagnosisInputFunction(
                trainingExamples,
                trainingTargets["compactness_mean"],
                numEpochs = 1,
                shuffle = False)
            predictValidationInputFunction = lambda: diagnosisInputFunction(
                validationExamples, validationTargets["compactness_Mean"],
                numEpochs = 1,
                shuffle = False)
        def recursive():
            print("Training Model")
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
        def userInput(
            cancerDataframeUI
            ):

            # User inputs get inputted here.

            radiusMeanUI = input("Please input the radius mean: ")
            textureMeanUI = input("Please input the texture mean: ")
            perimeterMeanUI = input("Please input perimeter mean: ")
            areaMeanUI = input("Please input area mean: ")
            smoothnessMeanUI = input("Please input smoothness mean: ")
            compactnessMeanUI = input("Please input compactness mean: ")
            concavityMeanUI = input("Please input concavity mean: ")
            concavePointsMeanUI = input("Please input concave points mean: ")
            symmetryMeanUI = input("Please input symmetry mean: ")
            fractalDimensionUI = input("Please input fractal dimension mean: ")

            # Setting up the user input array.

            userInputDataframe = [radiusMeanUI, textureMeanUI, perimeterMeanUI, areaMeanUI, smoothnessMeanUI, compactnessMeanUI, concavityMeanUI, concaveMeanUI, symmetryMeanUI, fractalDimensionUI] 

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
            if wrongUserInputCheck == Yes:
            
                if wrongUserInputCheck == No:
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

exit(0)


if __name__ == "__main__":
    tf.app.run()
