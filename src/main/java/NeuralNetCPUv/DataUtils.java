package NeuralNetCPUv;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class DataUtils {

    public static List<String[]> readCSV(String resourcePath) {
        List<String[]> rows = new ArrayList<>();
        InputStream is = DataUtils.class.getClassLoader().getResourceAsStream(resourcePath);
        if (is == null) throw new RuntimeException("File not found in resources: " + resourcePath);

        try (BufferedReader br = new BufferedReader(new InputStreamReader(is))) {
            String line;
            boolean first = true;
            while ((line = br.readLine()) != null) {
                if (first) { first = false; continue; } // skip header
                String[] values = line.split(",");
                rows.add(values);
            }
        } catch (IOException e) {
            throw new RuntimeException("Error reading CSV: " + e.getMessage(), e);
        }
        return rows;
    }

    public static void shuffleDataset(Matrix features, Matrix labels) {
        int n = features.data.length;
        List<Integer> indices = new ArrayList<>(n);
        for (int i = 0; i < n; i++) indices.add(i);
        Collections.shuffle(indices);

        float[][] newFeatures = new float[n][features.cols];
        float[][] newLabels = new float[n][labels.cols];
        for (int i = 0; i < n; i++) {
            newFeatures[i] = features.data[indices.get(i)].clone();
            newLabels[i] = labels.data[indices.get(i)].clone();
        }
        features.data = newFeatures;
        labels.data = newLabels;
    }

    public static void splitDataset(String resourcePath, int numTrain, int numTest,
                                    TrainDataset train, TestDataset test) {
        List<float[]> featuresList = new ArrayList<>();
        List<float[]> labelsList = new ArrayList<>();

        InputStream is = DataUtils.class.getClassLoader().getResourceAsStream(resourcePath);
        if (is == null) throw new RuntimeException("File not found in resources: " + resourcePath);

        try (BufferedReader br = new BufferedReader(new InputStreamReader(is))) {
            String line;
            boolean first = true;
            while ((line = br.readLine()) != null) {
                if (first) { first = false; continue; } // skip header
                String[] tokens = line.split(",");
                if (tokens.length < 5) continue;

                float[] featureRow = new float[4];
                for (int i = 0; i < 4; i++) featureRow[i] = Float.parseFloat(tokens[i]);

                float[] labelRow = new float[3];
                String lab = tokens[4].trim().toLowerCase();
                switch (lab) {
                    case "setosa" -> labelRow[0] = 1f;
                    case "versicolor" -> labelRow[1] = 1f;
                    case "virginica" -> labelRow[2] = 1f;
                    default -> {}
                }

                featuresList.add(featureRow);
                labelsList.add(labelRow);
            }
        } catch (IOException e) {
            throw new RuntimeException("Error reading dataset: " + e.getMessage(), e);
        }

        int total = featuresList.size();
        if (numTrain + numTest > total) {
            throw new IllegalArgumentException("Requested train+test > available samples (" + total + ")");
        }

        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < total; i++) indices.add(i);
        Collections.shuffle(indices);

        for (int i = 0; i < numTrain; i++) {
            train.features.data[i] = featuresList.get(indices.get(i)).clone();
            train.labels.data[i] = labelsList.get(indices.get(i)).clone();
        }
        for (int i = 0; i < numTest; i++) {
            test.features.data[i] = featuresList.get(indices.get(numTrain + i)).clone();
            test.labels.data[i] = labelsList.get(indices.get(numTrain + i)).clone();
        }
    }
}
