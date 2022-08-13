package network;

import data.DataPoint;

/**
 * Overall network class
 * @version 0.1
 * @author lucas
 */
public class NeuralNetwork {

    // All of the layers of the network (discluding input layer)
    private final Layer[] layers;

    /**
     *
     * @param layerSizes A list of numbers representing the size of each of the layers (in order)
     */
    public NeuralNetwork(int... layerSizes) {
        // Creates the layers
        layers = new Layer[layerSizes.length-1];
        for (int i = 0; i < layers.length; i++) {
            layers[i] = new Layer(layerSizes[i], layerSizes[i+1]);
        }

    }

    /**
     * Runs the inputs through the network
     * @param inputs The inputs to the network
     * @return Outputs from the network
     */
    private double[] calculateOutputs(double[] inputs) {
        assert inputs.length == layers[0].getNumNodesIn();

        for(Layer layer : layers) {
            inputs = layer.calculateOutputs(inputs);
        }
        return inputs;
    }

    /**
     * Classifies the input using the neural network
     * @param inputs Inputs to the neural network
     * @return The classification index of the input
     */
    private int classify(double[] inputs) {
        double[] outputs = calculateOutputs(inputs);

        int maxValueIndex = 0;
        double maxValue = outputs[0];
        for(int i = 1; i < outputs.length; i++) {
            if (outputs[i] > maxValue) {
                maxValue = outputs[i];
                maxValueIndex = i;
            }
        }

        return maxValueIndex;
    }

    /**
     * Calculates the cost of the overall network on a data point
     * @param dataPoint
     * @return
     */
    private double cost(DataPoint dataPoint) {
        double[] outputs = calculateOutputs(dataPoint.inputs);
        Layer outputLayer = layers[layers.length-1];
        double cost = 0;
        for (int outNode = 0; outNode < outputs.length; outNode++) {
            cost += outputLayer.nodeCost(outputs[outNode], dataPoint.expectedOutputs[outNode]);
        }

        return cost;
    }

    /**
     * Calculates the average cost of the network
     * @param data All data points in the set
     * @return Average cost of the network
     */
    private double cost(DataPoint[] data) {
        double totalCost = 0;

        for(DataPoint dp : data) {
            totalCost += cost(dp);
        }

        return totalCost / data.length;
    }


}
