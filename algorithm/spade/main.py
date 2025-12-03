import asyncio
import spade
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade import wait_until_finished
import os
from dotenv import load_dotenv


class WallEAgent(Agent):
    class MoveRightBehaviour(CyclicBehaviour):
        async def on_start(self):
            # Parámetros del mundo
            self.world_width = 5
            self.world_height = 5


            # Estado interno del agente
            self.pos = [1, 3]      # posición inicial (x, y)
            self.goal = (5, 3)     # meta


            print(f"[Wall-E] Iniciando en {tuple(self.pos)}, meta {self.goal}\n")


        async def run(self):
            x, y = self.pos
            print(f"[Wall-E] Posición actual: ({x}, {y})")
            # si llegamos
            if (x, y) == self.goal:
                print(f"[Wall-E] Meta alcanzada en {self.goal}.\n")
                # Marcamos el comportamiento para terminar en esta iteración
                self.kill(exit_code="goal_reached")
                return


            # Regla reactiva simple: si estoy en la fila 3 y x < 5, me muevo a la derecha
            if y == 3 and x < self.world_width:
                new_x = x + 1
                new_y = y


                # Verificar que el movimiento está dentro de la malla
                if 1 <= new_x <= self.world_width and 1 <= new_y <= self.world_height:
                    self.pos = [new_x, new_y]
                    print(f"[Wall-E] Me moví a {tuple(self.pos)}\n")
                else:
                    print("[Wall-E] Movimiento inválido, fuera de la malla.")
                    self.kill(exit_code="out_of_bounds")
                    return
            else:
                print(f"[Wall-E] No hay regla aplicable desde {tuple(self.pos)}.")
                self.kill(exit_code="no_rule")
                return


            # Pausa para que el movimiento se vea paso a paso
            await asyncio.sleep(1)


        async def on_end(self):
            print(f"[Wall-E] Behaviour terminado con exit_code = {self.exit_code}")
            # Detenemos el agente cuando acaba el comportamiento principal
            await self.agent.stop()


    async def setup(self):
        print(f"[Wall-E] Agente inicializado con JID {self.jid}")
        behav = self.MoveRightBehaviour()
        # Guardamos referencia por si luego quieres inspeccionar exit_code o estado
        self.move_behav = behav
        self.add_behaviour(behav)


async def main():
    load_dotenv()

    jid = os.getenv("JID")
    password = os.getenv("PASSWORD")

    wall_e = WallEAgent(jid, password)

    await wall_e.start(auto_register=True)
    print("[Wall-E] Iniciado. Ejecutando comportamiento...\n")

    # Esperar a que el agente termine (cuando el behaviour haga stop)
    await wait_until_finished(wall_e)
    print("[Wall-E] Agente finalizado.")


if __name__ == "__main__":
    # spade.run se encarga del loop asíncrono de forma estandar
    spade.run(main())
