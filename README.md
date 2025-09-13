# NeuralNet4J

This project presents a machine learning library focused on **simple and easy-to-use neural network training and inference in Java**.  
It provides a CPU-based implementation written from scratch, making it a lightweight and educational framework for beginners and a flexible base for advanced users.  

---

## Features
- **Cross-platform compatibility**: Runs on any platform with a Java Runtime Environment (JRE).
- **Pure Java implementation**: No external ML libraries required.
- **Ease of use**: Straightforward API for creating, training, and evaluating neural networks.
- **Training support**: Includes backpropagation and cross-entropy loss.
- **Customizability**: Adjustable network architecture, learning rates, and epochs.
- **Dataset utilities**: Shuffling, normalization, and dataset splitting.
- **Sample dataset**: Includes an example using the Iris dataset for classification.

## Project Structure

NeuralNet4J/
 └── src/
     └── main/
         └── java/
             └── NeuralNetCPUv/
                 ├── Activations.java
                 ├── DataUtils.java
                 ├── Layer.java
                 ├── Main.java
                 ├── Matrix.java
                 ├── NeuralNetwork.java
                 ├── RandomUtil.java
                 ├── TestDataset.java
                 ├── TrainDataset.java
                 ├── Trainer.java
                 └── Vector.java


### NeuralNetCPUv Contents:
- `Activations.java` – Activation functions (ReLU, Softmax, etc.)
- `DataUtils.java` – Dataset utilities (shuffle, normalize, split)
- `Layer.java` – Fully connected layer implementation
- `Main.java` – Entry point (demo on Iris dataset)
- `Matrix.java` – Matrix operations
- `NeuralNetwork.java` – Core neural network logic
- `RandomUtil.java` – Random number/shuffling utility
- `TestDataset.java` – Test dataset wrapper
- `TrainDataset.java` – Training dataset wrapper
- `Trainer.java` – Training loop (epochs, loss, evaluation)
- `Vector.java` – Vector operations
