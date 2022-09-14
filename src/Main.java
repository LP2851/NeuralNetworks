import data.BatchHandler;
import data.DataPoint;
import mnist.MnistReader;
import output.NetworkJSONExporter;

import java.io.IOException;

/**
 *
 * @version 0.1
 * @author lucas
 */
public class Main {

    private static DataPoint[] trainingData, testingData;
    private static final double learningRate = 0.25d;

    /**
     * Runs the neural network
     * @param args
     */
    public static void main(String[] args) {
        // Loading the datasets from the files.
        loadDataSets();

        // Displaying the images loaded (for debugging purposes)

//        try {
//            MnistPanel.showImages(trainingData);
//        } catch (Exception e) {
//            throw new RuntimeException(e);
//        }

        network.NeuralNetwork nn = new network.NeuralNetwork(28 * 28, 10, 10);
        //BatchHandler<DataPoint> batchHandler = new BatchHandler<>(1000, trainingData, false, 100);
        BatchHandler<DataPoint> batchHandler = new BatchHandler<>(1000, trainingData, 1, true);
        var batch = batchHandler.nextBatch();

        while (batch.length > 0) {
            nn.learn(batch, learningRate);

            System.out.print("Batch Number: " + batchHandler.getBatchNumber());
            System.out.print(" / " + batchHandler.calculateTotalBatchNumber() + "\n");
            System.out.println(nn.cost(testingData));
            batch = batchHandler.nextBatch();
        }

        if (NetworkJSONExporter.saveNetwork(nn, "network1"))
            System.out.println("Network Saved");
    }

    /**
     * Loads the dataset from the files.
     */
    private static void loadDataSets() {
        try {
            trainingData = MnistReader.getTrainingDataPoints();
            testingData = MnistReader.getTestingDataPoints();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}