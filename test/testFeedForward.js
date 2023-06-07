import { Feedforward } from 'neura_ai';

// XOR
var ff = new Feedforward([3], 2, 1);

for(var i = 0; i != 90; i++)
{
  ff.train(new Float32Array([1^1]), 0.05, new Float32Array([1, 1]));
  ff.train(new Float32Array([0^0]), 0.05, new Float32Array([0, 0]));
  ff.train(new Float32Array([0^1]), 0.05, new Float32Array([0, 1]));
  ff.train(new Float32Array([1^0]), 0.05, new Float32Array([1, 0]));
};

console.log(ff.run(new Float32Array([0, 1]));
console.log(ff.run(new Float32Array([1, 1]));
