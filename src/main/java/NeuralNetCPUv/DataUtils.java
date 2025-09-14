package NeuralNetCPUv;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Utility class for handling dataset operations such as:
 * - Reading CSV files from resources
 * - Shuffling datasets
 * - Splitting data into training and testing sets
 */
public class DataUtils {

    /**
     * Reads a CSV file from the resources folder.
     * Assumes the first row is a header and skips it.
     *
     * @param resourcePath path to the CSV file inside resources
     * @return a list of rows, where each row is represented as a String array
     */
    public static List<String[]> readCSV(String resourcePath) {
        List<String[]> rows = new ArrayList<>();

        // Step 1: Load file from resources using class loader
        InputStream inputStream = DataUtils.class.getClassLoader().getResourceAsStream(resourcePath);
        if (inputStream == null) {
            throw new RuntimeException("File not found in resources: " + resourcePath);
        }

        // Step 2: Read file line by line
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream))) {
            String line;
            boolean firstLine = true;

            while ((line = reader.readLine()) != null) {
                if (firstLine) {
                    // Skip header line
                    firstLine = false;
                    continue;
                }

                // Split by comma into tokens
                String[] values = line.split(",");
                rows.add(values);
            }
        } catch (IOException e) {
            throw new RuntimeException("Error reading CSV: " + e.getMessage(), e);
        }

        return rows;
    }

    /**
     * Shuffles dataset (features and labels together).
     *
     * @param features Matrix containing feature values
     * @param labels   Matrix containing label values
     */
    public static void shuffleDataset(Matrix features, Matrix labels) {
        int numSamples = features.data.length;

        // Step 1: Create a list of indices 0..n-1
        List<Integer> indices = new ArrayList<>(numSamples);
        for (int i = 0; i < numSamples; i++) {
            indices.add(i);
        }

        // Step 2: Shuffle indices randomly
        Collections.shuffle(indices);

        // Step 3: Create new shuffled matrices
        float[][] shuffledFeatures = new float[numSamples][features.cols];
        float[][] shuffledLabels = new float[numSamples][labels.cols];

        for (int i = 0; i < numSamples; i++) {
            int oldIndex = indices.get(i);
            shuffledFeatures[i] = features.data[oldIndex].clone();
            shuffledLabels[i] = labels.data[oldIndex].clone();
        }

        // Step 4: Replace old data with shuffled data
        features.data = shuffledFeatures;
        labels.data = shuffledLabels;
    }

    /**
     * Splits a dataset from CSV into training and testing sets.
     * Assumes dataset has 4 feature columns and 1 label column.
     * Label is categorical (setosa, versicolor, virginica).
     *
     * @param resourcePath path to CSV inside resources
     * @param numTrain     number of training samples
     * @param numTest      number of testing samples
     * @param train        TrainDataset object to fill
     * @param test         TestDataset object to fill
     */
    public static void splitDataset(String resourcePath,
                                    int numTrain,
                                    int numTest,
                                    TrainDataset train,
                                    TestDataset test) {
        List<float[]> featuresList = new ArrayList<>();
        List<float[]> labelsList = new ArrayList<>();

        // Step 1: Load file from resources
        InputStream inputStream = DataUtils.class.getClassLoader().getResourceAsStream(resourcePath);
        if (inputStream == null) {
            throw new RuntimeException("File not found in resources: " + resourcePath);
        }

        // Step 2: Read file line by line
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream))) {
            String line;
            boolean firstLine = true;

            while ((line = reader.readLine()) != null) {
                if (firstLine) {
                    // Skip header row
                    firstLine = false;
                    continue;
                }

                // Split row into tokens
                String[] tokens = line.split(",");

                // Skip if row does not have enough columns
                if (tokens.length < 5) {
                    continue;
                }

                // Step 3: Parse 4 feature values
                float[] featureRow = new float[4];
                for (int i = 0; i < 4; i++) {
                    featureRow[i] = Float.parseFloat(tokens[i]);
                }

                // Step 4: Parse label (convert string to one-hot vector of size 3)
                float[] labelRow = new float[3];
                String labelText = tokens[4].trim().toLowerCase();

                if (labelText.equals("setosa")) {
                    labelRow[0] = 1.0f;
                } else if (labelText.equals("versicolor")) {
                    labelRow[1] = 1.0f;
                } else if (labelText.equals("virginica")) {
                    labelRow[2] = 1.0f;
                }
                // else ignore unrecognized labels

                featuresList.add(featureRow);
                labelsList.add(labelRow);
            }
        } catch (IOException e) {
            throw new RuntimeException("Error reading dataset: " + e.getMessage(), e);
        }

        // Step 5: Ensure enough data exists
        int totalSamples = featuresList.size();
        if (numTrain + numTest > totalSamples) {
            throw new IllegalArgumentException(
                    "Requested train+test > available samples (" + totalSamples + ")"
            );
        }

        // Step 6: Shuffle indices for randomness
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < totalSamples; i++) {
            indices.add(i);
        }
        Collections.shuffle(indices);

        // Step 7: Fill training dataset
        for (int i = 0; i < numTrain; i++) {
            int idx = indices.get(i);
            train.features.data[i] = featuresList.get(idx).clone();
            train.labels.data[i] = labelsList.get(idx).clone();
        }

        // Step 8: Fill testing dataset
        for (int i = 0; i < numTest; i++) {
            int idx = indices.get(numTrain + i);
            test.features.data[i] = featuresList.get(idx).clone();
            test.labels.data[i] = labelsList.get(idx).clone();
        }
    }
}
