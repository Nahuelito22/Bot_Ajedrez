import { Chess } from 'chess.js';

const chess = new Chess();
const moves = chess.moves({ square: 'e2', verbose: true });
console.log('Moves for e2:', moves);

