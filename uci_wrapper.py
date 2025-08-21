# uci_wrapper.py

import sys
import requests
import chess

# --- CONFIGURACIÓN ---
# La dirección de tu API de FastAPI que debe estar corriendo localmente
API_URL = "http://127.0.0.1:8000/predict_move"
ENGINE_NAME = "BotAjedrezLSTM_Nahu"
ENGINE_AUTHOR = "Matias Nahuel Ghilardi Salinas"
# --------------------

def send_uci_response(message):
    """Envía un mensaje a la interfaz gráfica (Cute Chess)."""
    sys.stdout.write(message + "\n")
    sys.stdout.flush()

def main_loop():
    """Bucle principal que escucha los comandos UCI."""
    board = chess.Board()

    while True:
        line = sys.stdin.readline().strip()
        if not line:
            continue

        parts = line.split()
        command = parts[0]

        if command == "uci":
            send_uci_response(f"id name {ENGINE_NAME}")
            send_uci_response(f"id author {ENGINE_AUTHOR}")
            send_uci_response("uciok")
        
        elif command == "isready":
            send_uci_response("readyok")

        elif command == "ucinewgame":
            board.reset()

        elif command == "position":
            # Formato: position startpos moves e2e4 e7e5 ...
            # Formato: position fen <fen_string> moves e2e4 ...
            board.reset()
            moves_start_index = -1
            if "startpos" in parts:
                moves_start_index = parts.index("startpos") + 2
            elif "fen" in parts:
                fen_parts = parts[parts.index("fen") + 1 : parts.index("moves") if "moves" in parts else len(parts)]
                board.set_fen(" ".join(fen_parts))
                if "moves" in parts:
                    moves_start_index = parts.index("moves") + 1

            if moves_start_index != -1:
                for move_uci in parts[moves_start_index:]:
                    try:
                        board.push_uci(move_uci)
                    except ValueError:
                        pass # Ignorar movimientos ilegales en la línea de comandos

        elif command == "go":
            # Es nuestro turno de pensar.
            # Convertimos el historial a notación SAN, que es lo que nuestra API espera.
            history_san = [board.san(move) for move in board.move_stack]

            try:
                # Hacemos la petición a nuestra API
                response = requests.post(API_URL, json={"moves": history_san})
                response.raise_for_status() # Lanza un error si la petición falla
                
                # Recibimos la lista de jugadas propuestas por el bot
                suggested_moves = response.json().get("bot_moves", [])
                
                best_legal_move = None
                # Probamos cada jugada hasta encontrar una legal
                for move_san in suggested_moves:
                    try:
                        move = board.parse_san(move_san)
                        if move in board.legal_moves:
                            best_legal_move = move
                            break
                    except ValueError:
                        # La jugada no es válida en formato SAN
                        continue
                
                # Si ninguna jugada de la IA fue legal, jugamos al azar como Plan C
                if not best_legal_move:
                    best_legal_move = list(board.legal_moves)[0]

                send_uci_response(f"bestmove {best_legal_move.uci()}")

            except requests.exceptions.RequestException as e:
                # Si la API falla, jugamos al azar para no perder la partida
                move = list(board.legal_moves)[0]
                send_uci_response(f"bestmove {move.uci()}")

        elif command == "quit":
            break

if __name__ == "__main__":
    main_loop()