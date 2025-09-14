package NeuralNetCPUv;

/**
 * Training dataset representation.
 * Stores input features, labels (expected outputs),
 * and provides utilities like printing and shuffling.
 */
public class TrainDataset {
    public Matrix features;
    public Matrix labels;    
    public int numExamples;

    /**
     * Constructor to create a training dataset with preallocated feature and label matrices.
     *
     * @param numExamples number of examples in the dataset
     * @param inputSize   number of input features per example
     * @param outputSize  number of output labels per example
     */
    public TrainDataset(int numExamples, int inputSize, int outputSize) {
        this.numExamples = numExamples;
        this.features = new Matrix(numExamples, inputSize);
        this.labels = new Matrix(numExamples, outputSize);
    }

    /**
     * Default constructor creates an empty dataset.
     */
    public TrainDataset() {
        this.numExamples = 0;
        this.features = null;
        this.labels = null;
    }

    /**
     * Prints one sample from the dataset (both features and labels).
     *
     * @param index sample index to print
     */
    public void printSample(int index) {
        // Check if index is invalid (less than 0)
        if (index < 0) {
            System.out.println("Index out of bounds! Index cannot be negative.");
            return;
        }

        // Check if index is invalid (greater or equal to total examples)
        if (index >= numExamples) {
            System.out.println("Index out of bounds! Index is greater than dataset size.");
            return;
        }

        // If index is valid, print the feature vector and label vector
        System.out.print("Features: ");
        Vector.print(features.data[index]);

        System.out.print("Labels: ");
        Vector.print(labels.data[index]);
    }

    /**
     * Shuffles the dataset so inputs and outputs remain aligned.
     */
    public void shuffle() {
        RandomUtil.shuffle(this.features.data, this.labels.data);
    }
}

