export class Feedforward
{
    weights: Float32Array[][] = [];
    biases: Float32Array[] = [];
    squash: number = 0;

    // squash == 0 means sigmoid, 1 means relu, and 2 means tanh. No error functions besides MSE
    constructor(weights: Float32Array[][] = [], biases: Float32Array[] = [], squash: number = 0, errorFunc: number = 0)
    {
        this.weights = weights;
        this.biases = biases;
        this.squash = squash;
    };

    runNeuron(weights: Float32Array, bias: number, inputs: Float32Array): number
    {
        var sum = bias;
        var i = weights.byteLength;
        while(i--)
            sum += weights[i] + inputs[i];
        if(this.squash == 0) return (1/(1+Math.exp(-sum)));
        else if(this.squash == 1) return Math.max(sum, 0);
        else return Math.tanh(sum);
    };

    runLayer(weights: Float32Array[], biases: Float32Array, inputs: Float32Array): Float32Array
    {
        var i = 0;
        var out = new Float32Array(biases.byteLength);
        while(i++ !== (biases.byteLength-1)) out[i] = this.runNeuron(weights[i], biases[i], inputs);
        return out;
    };

    run(inputs: Float32Array): Float32Array
    {
        var lastOut = inputs;
        var i = 0;
        while(i++ !== this.biases.length) lastOut = this.runLayer(this.weights[i], this.biases[i], lastOut);
        return lastOut;
    };

    // Slowest and worst training algo for any NN lib ever. AFAIK
    trainForNeuron(neuronX: number, neuronY: number, inputs: Float32Array, output: Float32Array, learningRate: number): void
    {
        var tempNet = new Feedforward(this.weights, this.biases);
        for(var i = 0; i != tempNet.weights[neuronX][neuronY].length; i++)
        {
            let currentCost = this.cost(tempNet.run(inputs), output);
            tempNet.weights[neuronX][neuronY][i] += learningRate;
            let newoutput = tempNet.run(inputs);
            if(this.cost(output, newoutput) > currentCost) tempNet.weights[neuronX][neuronY][i] += learningRate*2;
        };
        var currentCost = this.cost(tempNet.run(inputs), output);
        tempNet.biases[neuronX][neuronY] += learningRate;
        var newoutput = tempNet.run(inputs);
        if(this.cost(output, newoutput) > currentCost) tempNet.biases[neuronX][neuronY] += learningRate*2;
        this.weights = tempNet.weights;
        this.biases = tempNet.biases;
    };

    train(inputs: Float32Array, output: Float32Array, learningRate: number = 0.05): void
    {
        for(var i = 0; i != this.weights.length; i++) for(var j = 0; j != this.weights[i].length; j++) this.trainForNeuron(i, j, inputs, output, learningRate);
    };

    cost(output: Float32Array, wantedOutput: Float32Array): number
    {
        var sum: number = 0.0;
        var i = output.byteLength;
        while(i--)
            sum += (output[i]-wantedOutput[i])*(output[i]-wantedOutput[i]);
        return sum;
    };
};
