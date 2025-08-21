# uci_wrapper.py (Versión Final Corregida y Robusta)

import sys
import requests
import chess
import logging

# --- CONFIGURACIÓN ---
API_URL = "http://127.0.0.1:8000/predict_move"
ENGINE_NAME = "BotAjedrezLSTM_Nahu"
ENGINE_AUTHOR = "Matias Nahuel Ghilardi Salinas"

# Configurar un log para depuración
logging.basicConfig(filename="uci_wrapper.log", level=logging.DEBUG, filemode='w')

def send_uci_response(message):
    """Envía un mensaje a la interfaz gráfica y lo registra en el log."""
    logging.debug(f"SENDING: {message}")
    sys.stdout.write(message + "\n")
    sys.stdout.flush()

def main_loop():
    """Bucle principal que escucha los comandos UCI."""
    board = chess.Board()
    history_san = [] # <<< Guardaremos el historial aquí

    while True:
        line = sys.stdin.readline().strip()
        logging.debug(f"RECEIVED: {line}")
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
            history_san = [] # <<< Limpiamos el historial para la nueva partida

        elif command == "position":
            # --- LÓGICA DE POSICIÓN CORREGIDA ---
            board.reset()
            history_san = [] # Limpiamos para reconstruir
            
            moves_start_index = -1
            if "startpos" in parts:
                if "moves" in parts:
                    moves_start_index = parts.index("moves") + 1
            
            if moves_start_index != -1:
                # Procesamos cada jugada, guardando su notación en el momento correcto
                for move_uci in parts[moves_start_index:]:
                    try:
                        move = chess.Move.from_uci(move_uci)
                        if move in board.legal_moves:
                            history_san.append(board.san(move))
                            board.push(move)
                    except ValueError:
                        pass
        
        elif command == "go":
            try:
                # Ahora simplemente usamos el historial que ya construimos
                response = requests.post(API_URL, json={"moves": history_san})
                response.raise_for_status()
                
                suggested_moves = response.json().get("bot_moves", [])
                best_legal_move = None
                
                for move_san in suggested_moves:
                    try:
                        move = board.parse_san(move_san)
                        if move in board.legal_moves:
                            best_legal_move = move
                            break
                    except ValueError:
                        continue
                
                if not best_legal_move and list(board.legal_moves):
                    best_legal_move = list(board.legal_moves)[0]
                
                if best_legal_move:
                    send_uci_response(f"bestmove {best_legal_move.uci()}")

            except Exception as e:
                logging.error(f"Error en el comando 'go': {e}")
                if list(board.legal_moves):
                    move = list(board.legal_moves)[0]
                    send_uci_response(f"bestmove {move.uci()}")

        elif command == "quit":
            logging.debug("Comando 'quit' recibido. Terminando.")
            break

if __name__ == "__main__":
    try:
        main_loop()
    except Exception as e:
        logging.error(f"Error fatal en el bucle principal: {e}")