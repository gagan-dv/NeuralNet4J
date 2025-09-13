package NeuralNetCPUv;

public class TestDataset {
    public Matrix features;
    public Matrix labels;
    public int numExamples;

    public TestDataset(int numExamples, int inputSize, int outputSize) {
        this.numExamples = numExamples;
        this.features = new Matrix(numExamples, inputSize);
        this.labels = new Matrix(numExamples, outputSize);
    }

    public TestDataset() {
        this.numExamples = 0;
        this.features = null;
        this.labels = null;
    }

    public void printSample(int index) {
        if (index < 0 || index >= numExamples) {
            System.out.println("Index out of bounds!");
            return;
        }
        System.out.print("Features: ");
        Vector.print(features.data[index]);
        System.out.print("Labels: ");
        Vector.print(labels.data[index]);
    }
}
