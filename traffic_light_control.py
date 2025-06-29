"""
DeepTrafficQ: A Deep Q-Learning (DQN) Traffic Signal Control System
Author: Albin James Maliakal
GitHub: https://albinjm.github.io/ 
Description: Uses Q-learning with a deep neural network to optimize traffic light phases 
             at a 4-way intersection, minimizing vehicle waiting time. Implements experience 
             replay and SUMO simulation for training.
"""

from __future__ import absolute_import
from __future__ import print_function
from sumolib import checkBinary

import sys
import random
import traci

from model import DQNAgent
from generator import SumoIntersection

if __name__ == '__main__':
    sumoInt = SumoIntersection()
    options = sumoInt.getOptions()

    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')
    sumoInt.routeFileGenerator()

    episodes = 100
    batch_size = 100

    green_duration = 10
    yellow_duration = 6
    agentGenerator = DQNAgent()
    try:
        agentGenerator.load('Models/reinf_traf_control.h5')
    except:
        print('No models found')

    for e in range(episodes):
        step = 0
        haltTime = 0
        reward1 = 0
        reward2 = 0
        netReward = 0.9*(reward1 - reward2)
        stepsCounter = 0
        action = 0

        traci.start([sumoBinary, "-c", "cross3ltl.sumocfg", '--start'])
        traci.trafficlight.setPhase("0", 0)
        traci.trafficlight.setPhaseDuration("0", 200)
        while traci.simulation.getMinExpectedNumber() > 0 and stepsCounter < 7000:
            traci.simulationStep()
            state = sumoInt.getState()
            action = agentGenerator.act(state)
            signalLight = state[2]

            if(action == 0 and signalLight[0][0][0] == 0):
                for i in range(6):
                    stepsCounter += 1
                    traci.trafficlight.setPhase('0', 1)
                    haltTime += (traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber(
                        '2si') + traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si'))
                    traci.simulationStep()
                for i in range(10):
                    stepsCounter += 1
                    traci.trafficlight.setPhase('0', 2)
                    haltTime += (traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber(
                        '2si') + traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si'))
                    traci.simulationStep()
                for i in range(6):
                    stepsCounter += 1
                    traci.trafficlight.setPhase('0', 3)
                    haltTime += (traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber(
                        '2si') + traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si'))
                    traci.simulationStep()

                reward1 = traci.edge.getLastStepVehicleNumber(
                    '1si') + traci.edge.getLastStepVehicleNumber('2si')
                reward2 = traci.edge.getLastStepHaltingNumber(
                    '3si') + traci.edge.getLastStepHaltingNumber('4si')
                for i in range(10):
                    stepsCounter += 1
                    traci.trafficlight.setPhase('0', 4)
                    reward1 += traci.edge.getLastStepVehicleNumber(
                        '1si') + traci.edge.getLastStepVehicleNumber('2si')
                    reward2 += traci.edge.getLastStepHaltingNumber(
                        '3si') + traci.edge.getLastStepHaltingNumber('4si')
                    haltTime += (traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber(
                        '2si') + traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si'))
                    traci.simulationStep()

            if(action == 0 and signalLight[0][0][0] == 1):
                reward1 = traci.edge.getLastStepVehicleNumber(
                    '1si') + traci.edge.getLastStepVehicleNumber('2si')
                reward2 = traci.edge.getLastStepHaltingNumber(
                    '3si') + traci.edge.getLastStepHaltingNumber('4si')
                for i in range(10):
                    stepsCounter += 1
                    traci.trafficlight.setPhase('0', 4)
                    reward1 += traci.edge.getLastStepVehicleNumber(
                        '1si') + traci.edge.getLastStepVehicleNumber('2si')
                    reward2 += traci.edge.getLastStepHaltingNumber(
                        '3si') + traci.edge.getLastStepHaltingNumber('4si')
                    haltTime += (traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber(
                        '2si') + traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si'))
                    traci.simulationStep()

            if(action == 1 and signalLight[0][0][0] == 0):
                reward1 = traci.edge.getLastStepVehicleNumber(
                    '4si') + traci.edge.getLastStepVehicleNumber('3si')
                reward2 = traci.edge.getLastStepHaltingNumber(
                    '2si') + traci.edge.getLastStepHaltingNumber('1si')
                for i in range(10):
                    stepsCounter += 1
                    reward1 += traci.edge.getLastStepVehicleNumber(
                        '4si') + traci.edge.getLastStepVehicleNumber('3si')
                    reward2 += traci.edge.getLastStepHaltingNumber(
                        '2si') + traci.edge.getLastStepHaltingNumber('1si')
                    traci.trafficlight.setPhase('0', 0)
                    haltTime += (traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber(
                        '2si') + traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si'))
                    traci.simulationStep()

            if(action == 1 and signalLight[0][0][0] == 1):
                for i in range(6):
                    stepsCounter += 1
                    traci.trafficlight.setPhase('0', 5)
                    haltTime += (traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber(
                        '2si') + traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si'))
                    traci.simulationStep()
                for i in range(10):
                    stepsCounter += 1
                    traci.trafficlight.setPhase('0', 6)
                    haltTime += (traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber(
                        '2si') + traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si'))
                    traci.simulationStep()
                for i in range(6):
                    stepsCounter += 1
                    traci.trafficlight.setPhase('0', 7)
                    haltTime += (traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber(
                        '2si') + traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si'))
                    traci.simulationStep()

                reward1 = traci.edge.getLastStepVehicleNumber(
                    '4si') + traci.edge.getLastStepVehicleNumber('3si')
                reward2 = traci.edge.getLastStepHaltingNumber(
                    '2si') + traci.edge.getLastStepHaltingNumber('1si')
                for i in range(10):
                    stepsCounter += 1
                    traci.trafficlight.setPhase('0', 0)
                    reward1 += traci.edge.getLastStepVehicleNumber(
                        '4si') + traci.edge.getLastStepVehicleNumber('3si')
                    reward2 += traci.edge.getLastStepHaltingNumber(
                        '2si') + traci.edge.getLastStepHaltingNumber('1si')
                    haltTime += (traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber(
                        '2si') + traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si'))
                    traci.simulationStep()

            generateNewState = sumoInt.getState()
            reward = reward1 - reward2
            agentGenerator.remember(
                state, action, reward, generateNewState, False)
            if(len(agentGenerator.memory) > batch_size):
                agentGenerator.replay(batch_size)

        mem = agentGenerator.memory[-1]
        del agentGenerator.memory[-1]
        agentGenerator.memory.append((mem[0], mem[1], reward, mem[3], True))
        print('episode - ' + str(e) + ' total waiting time - ' + str(haltTime))
        traci.close(wait=False)

sys.stdout.flush()
