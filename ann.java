import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

class EmotionDetection {
    public static void main(String[] args) {
        // Specify the paths of the training and test datasets
        String trainingDatasetPath = "C:\\visualize\\training_data.txt";
        String testDatasetPath = "C:\\visualize\\testing_data.txt";

        // Specify the number of classes, input size, and other parameters
        int numClasses = 6;
        int inputSize = 48 * 48;
        int numHiddenUnits = 10;
        double learningRate = 0.01;
        int numEpochs = 100;
        int numTrainingImages = 500;
        int numTestImages = 15;

        // Load the training dataset
        double[][] trainingInputs = new double[numTrainingImages][inputSize];
        int[] trainingTargets = new int[numTrainingImages];
        try {
            File trainingDatasetFile = new File(trainingDatasetPath);
            Scanner scanner = new Scanner(trainingDatasetFile);
            for (int i = 0; i < numTrainingImages; i++) {
                String line = scanner.nextLine();
                String[] values = line.split(" ");
                trainingTargets[i] = Integer.parseInt(values[0]);
                for (int j = 0; j < inputSize; j++)
                    trainingInputs[i][j] = Double.parseDouble(values[j + 1]) / 255;
            }
            scanner.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            return;
        }

        // Load the test dataset
        double[][] testInputs = new double[numTestImages][inputSize];
        int[] testTargets = new int[numTestImages];
        try {
            File testDatasetFile = new File(testDatasetPath);
            Scanner scanner = new Scanner(testDatasetFile);
            for (int i = 0; i < numTestImages; i++) {
                String line = scanner.nextLine();
                String[] values = line.split(" ");
                testTargets[i] = Integer.parseInt(values[0]);
                for (int j = 0; j < inputSize; j++)
                    testInputs[i][j] = Double.parseDouble(values[j + 1]) / 255;
            }
            scanner.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            return;
        }

        // Initialize the weights
        double[][] hiddenLayerWeights = new double[numHiddenUnits][inputSize + 1]; // +1 for bias weight
        double[][] outputLayerWeights = new double[numClasses][numHiddenUnits + 1]; // +1 for bias weight
        initializeWeights(hiddenLayerWeights);
        initializeWeights(outputLayerWeights);

        // Train the neural network
        trainNeuralNetwork(trainingInputs, trainingTargets, hiddenLayerWeights, outputLayerWeights, learningRate,
                numEpochs);
for(int i=0;i<hiddenLayerWeights.length;i++)
{
    for(int j=0;j<hiddenLayerWeights[0].length;j++)
        System.out.print(hiddenLayerWeights[i][j]);
    System.out.println();
}
        // Test the neural network and calculate accuracy
        double accuracy = testNeuralNetwork(testInputs, testTargets, hiddenLayerWeights, outputLayerWeights);
        System.out.println("Accuracy on test images: " + accuracy);
    }

    // Initialize the weights with random values
    private static void initializeWeights(double[][] weights) {
        int numRows = weights.length; // 4
        int numCols = weights[0].length; // 48*48+1
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                weights[i][j] = Math.random() * 0.5;
            }
        }
    }

    // Train the neural network
    private static void trainNeuralNetwork(double[][] inputs, int[] targets, double[][] hiddenLayerWeights,
                                           double[][] outputLayerWeights, double learningRate, int numEpochs) {
        int numImages = inputs.length;
        int numHiddenUnits = hiddenLayerWeights.length;
        int numClasses = outputLayerWeights.length;

        for (int epoch = 0; epoch < numEpochs; epoch++) {
            for (int image = 0; image < numImages; image++) {
                // Forward pass
                double[] hiddenLayerOutput = computeHiddenLayerOutput(inputs[image], hiddenLayerWeights);
                double[] outputLayerOutput = computeOutputLayerOutput(hiddenLayerOutput, outputLayerWeights);

                // Backward pass
                double[] outputLayerError = new double[numClasses];
                double[] hiddenLayerError = new double[numHiddenUnits];

                for (int i = 0; i < numClasses; i++) {
                    outputLayerError[i] = outputLayerOutput[i] * (1 - outputLayerOutput[i])
                            * (targets[image] - outputLayerOutput[i]);

                    for (int j = 0; j < numHiddenUnits; j++) {
                        outputLayerWeights[i][j] += learningRate * outputLayerError[i] * hiddenLayerOutput[j];
                    }
                    outputLayerWeights[i][numHiddenUnits] += learningRate * outputLayerError[i]; // Update bias weight
                }

                for (int i = 0; i < numHiddenUnits; i++) {
                    double sum = 0.0;
                    for (int j = 0; j < numClasses; j++) {
                        sum += outputLayerWeights[j][i] * outputLayerError[j];
                    }
                    hiddenLayerError[i] = hiddenLayerOutput[i] * (1 - hiddenLayerOutput[i]) * sum;

                    for (int j = 0; j < inputs[image].length; j++) {
                        hiddenLayerWeights[i][j] += learningRate * hiddenLayerError[i] * inputs[image][j];
                    }
                    hiddenLayerWeights[i][inputs[image].length] += learningRate * hiddenLayerError[i]; // Update bias weight
                }

                // Batch normalization
                hiddenLayerOutput = batchNormalize(hiddenLayerOutput);
            }
        }
    }

    // Compute the output of the hidden layer
    private static double[] computeHiddenLayerOutput(double[] input, double[][] weights) {
        int numHiddenUnits = weights.length;
        double[] output = new double[numHiddenUnits];

        for (int i = 0; i < numHiddenUnits; i++) {
            double sum = weights[i][input.length]; // Bias weight
            for (int j = 0; j < input.length; j++) {
                sum += weights[i][j] * input[j];
            }
            output[i] = sigmoid(sum); // Apply activation function to the sum
        }

        return output;
    }

    // Compute the output of the output layer
    private static double[] computeOutputLayerOutput(double[] input, double[][] weights) {
        int numClasses = weights.length;
        double[] output = new double[numClasses];
        for (int i = 0; i < numClasses; i++) {
            double sum = weights[i][input.length]; // Bias weight
            for (int j = 0; j < input.length; j++) {
                sum += weights[i][j] * input[j];
            }
            output[i] = sigmoid(sum);
        }
        return output;
    }

    // Test the neural network and calculate accuracy
    private static double testNeuralNetwork(double[][] inputs, int[] targets, double[][] hiddenLayerWeights,
                                            double[][] outputLayerWeights) {
        int numImages = inputs.length;
        int numCorrect = 0;

        for (int image = 0; image < numImages; image++) {
            double[] hiddenLayerOutput = computeHiddenLayerOutput(inputs[image], hiddenLayerWeights);
            double[] outputLayerOutput = computeOutputLayerOutput(hiddenLayerOutput, outputLayerWeights);

            int predictedClass = getPredictedClass(outputLayerOutput);
          //  System.out.println(predictedClass);
            if (predictedClass == targets[image]) {
                numCorrect++;
            }
        }

        return (double) numCorrect / numImages;
    }

    // Apply the sigmoid activation function
    private static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    // Get the predicted class based on the output layer output
    private static int getPredictedClass(double[] outputLayerOutput) {
        int numClasses = outputLayerOutput.length;
        int predictedClass = 0;
        double maxOutput = outputLayerOutput[0];

        for (int i = 1; i < numClasses; i++) {
            if (outputLayerOutput[i] > maxOutput) {
                predictedClass = i;
                maxOutput = outputLayerOutput[i];
            }
        }

        return predictedClass;
    }

    // Apply batch normalization to the hidden layer output
    private static double[] batchNormalize(double[] input) {
        double mean = 0.0;
        double variance = 0.0;

        for (int i = 0; i < input.length; i++) {
            mean += input[i];
        }
        mean /= input.length;

        for (int i = 0; i < input.length; i++) {
            variance += Math.pow(input[i] - mean, 2);
        }
        variance /= input.length;

        for (int i = 0; i < input.length; i++) {
            input[i] = (input[i] - mean) / Math.sqrt(variance + 1e-8);
        }

        return input;
    }
}
