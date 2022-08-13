public class Layer {

    private final int numNodesIn, numNodesOut;

    private final double[][] weights;
    private final double[] biases;

    public Layer(int numNodesIn, int numNodesOut) {
        this.numNodesIn = numNodesIn;
        this.numNodesOut = numNodesOut;

        weights = new double[numNodesIn][numNodesOut];
        biases = new double[numNodesOut];
    }

    public double[] calculateOutputs(double[] inputs) {
        double[] activations = new double[numNodesOut];

        for (int outNode = 0; outNode < numNodesOut; outNode++) {
            double weightedInput = biases[outNode];
            for (int inNode = 0; inNode < numNodesIn; inNode++) {
                weightedInput += inputs[inNode] * weights[inNode][outNode];
            }
            activations[outNode] = sigmoidFunction(weightedInput);
        }

        return activations;
    }


    public double nodeCost(double outputActivation, double expectedOutput) {
        double error = outputActivation - expectedOutput;
        return error * error;
    }

    public int getNumNodesIn() {
        return numNodesIn;
    }


    // ACTIVATION FUNCTIONS

    private double stepFunction(double x) {
        return (x > 0) ? 1 : 0;
    }

    private double sigmoidFunction(double x) {
        return 1d / (1d + Math.exp(-x));
    }

    private double hyperbolicTangentFunction(double x) {
        double e2w = Math.exp(2 * x);
        return (e2w - 1) / (e2w + 1);
    }

    /**
     * SiLU
     * @param x
     * @return
     */
    private double siLU(double x) {
        return x / (1 + Math.exp(-x));
    }
    /**
     * ReLU
     * @param x
     * @return
     */
    private double reLU(double x) {
        return Math.max(0, x);
    }
}
