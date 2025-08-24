
const config = {
    API_URL: window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
        ? 'http://127.0.0.1:8000/predict_move'
        : 'https://nahuelito22-bot-ajedrez.hf.space/predict_move'
};
