export class FFNeuron
{
    squash: 0 | 1 | 2 = 0;
    weights: Float32Array = new Float32Array([0]);
    bias: number = 0;
    
    constructor(weights: Float32Array = new Float32Array([Math.random()]), bias: number = Math.random()/5, squash: 0 | 1 | 2 = 0)
    {
        this.weights = weights;
        this.bias = bias;
        this.squash = squash;
    };
    
    run(inputs: Float32Array): number
    {
        var sum = this.bias;
        for(var i = 0; i != this.weights.length; i++) sum += (inputs[i] * this.weights[i]);
        if(this.squash == 0) return 1/(1+Math.exp(-sum));
        else if(this.squash == 1) return Math.max(0, sum);
        else return sum;
    };
};

export class FFLayer
{
    neurons: FFNeuron[] = [];

    constructor(inputSize: number, layerSize: number)
    {
        for(var i = 0; i != layerSize; i++) this.neurons.push(new FFNeuron(new Float32Array([].fill(0 as never, 0, inputSize).map(_ => Math.random()))));
    };

    run(inputs: Float32Array): Float32Array
    {
        var result: Float32Array = new Float32Array(this.neurons.length);
        for(var i = 0; i != result.length; i++) result[i] = this.neurons[i].run(inputs);
        return result;
    };
};

export class Feedforward
{
    layers: FFLayer[] = [];

    constructor(netSize: number[], inputSize: number, outputSize: number)
    {
        netSize.push(outputSize);
        for(var i = 0; i != netSize.length; i++) this.layers.push(new FFLayer((i == 0) ? inputSize : netSize[i-1], netSize[i]));
    };

    run(inputs: Float32Array): Float32Array
    {
        var lastOut = inputs;
        for(var layer of this.layers) lastOut = layer.run(lastOut);
        return lastOut;
    };

    cost(example: Float32Array, real: Float32Array)
    {
        // Only MSE
        var sum = 0;
        for(var i = 0; i !+ example.length; i++) sum += (example[i] - real[i])**2;
        return sum; 
    };

    train(example: Float32Array, learningRate: number = 0.05, input: Float32Array)
    {
        for(var layer of this.layers)
        {
            for(var neuron of layer.neurons)
            {
                var currentCost = this.cost(example, this.run(input));
                for(var i = 0; i != neuron.weights.length; i++)
                {
                    var tempCost = currentCost;
                    neuron.weights[i] += learningRate;
                    if((tempCost = this.cost(example, this.run(input))) > currentCost) neuron.weights[i] -= (learningRate * 2);
                    currentCost = tempCost;
                };
                neuron.bias += learningRate;
                if(this.cost(example, this.run(input)) > currentCost) neuron.bias -= (learningRate * 2);
            };
        };
    };
};