export class Context
{
    module: WebAssembly.Module;
    instance: WebAssembly.Instance;

    constructor(ab: ArrayBuffer)
    {
        this.module = new WebAssembly.Module(ab);
        this.instance = new WebAssembly.Instance(this.module);
    };

    static async loadFromURL(url: string): Promise<Context>
    {
        var req = await fetch(url);
        var ab = await req.arrayBuffer();
        return new Context(ab);
    };
};

export class Feedforward
{
    context: Context;
    weights: Float32Array[][] = [];
    biases: Float32Array[] = [];
    squash: number = 0;

    constructor(context: Context, weights: Float32Array[][] = [], biases: Float32Array[] = [], squash: number = 0)
    {
        this.context = context;
        this.weights = weights;
        this.biases = biases;
        this.squash = squash;
    };

    run(inputs: Float32Array): Float32Array
    {
        (this.context.instance.exports.clearArray1FF as Function)();
        (this.context.instance.exports.clearArray2FF as Function)();
        (this.context.instance.exports.clearArray3FF as Function)();
        (this.context.instance.exports.clearArray4FF as Function)();
        this.setData();
        (this.context.instance.exports.runFeedforward as Function)(this.squash, this.weights[0].length);
        return this.getArray1();
    };

    setData(): void
    {
        var i = 0;
        while((i++) != this.biases.length)
        {
            (this.context.instance.exports.setArray3FF as Function)(this.biases[i].length);
            var j = 0;
            while((j++) != this.biases[i].length) 
            {
                var k = 0;
                (this.context.instance.exports.setArray2FF as Function)(this.biases[i][j]);
                while((k++) != (this.biases[i-1].length)) (this.context.instance.exports.setArray1FF as Function)(this.weights[i][j][k]);
            };
        };
    };

    getArray1(): Float32Array
    {
        var out = new Float32Array((this.context.instance.exports.lengthOfArray1FF as Function)());
        var i = out.length;
        while(i--)
            out[i] = (this.context.instance.exports.getFromArray1FF as Function)(i);
        return out;
    }
};