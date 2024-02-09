import torch
import random
import numpy as np
from snake import SnakeGameAI,Direction,Point
from collections import deque
from model import Linear_QNet,QTrainer
from helper import plot

Max_Memory = 100_000
Batch_Size =1000
lr = 0.01

class Agent:
    def __init__(self):
        self.num_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=Max_Memory)
        self.model = Linear_QNet(11,256,3)
        self.trainer = QTrainer(self.model,lr=lr,gamma=self.gamma)     

    def get_state(self,snake):
        head = snake.snake[0]
        point_l = Point(head.x-20,head.y)
        point_r = Point(head.x+20,head.y)
        point_u = Point(head.x,head.y-20)
        point_d = Point(head.x,head.y+20)

        dir_l = snake.direction == Direction.LEFT
        dir_r = snake.direction == Direction.RIGHT
        dir_u = snake.direction == Direction.UP
        dir_d = snake.direction == Direction.DOWN

        state = [
            #danger in moving straight
            (dir_r and snake.is_collision(point_r))or
            (dir_l and snake.is_collision(point_l))or
            (dir_u and snake.is_collision(point_u))or
            (dir_d and snake.is_collision(point_d)),

            #danger in moving right
            (dir_u and snake.is_collision(point_r))or
            (dir_d and snake.is_collision(point_l))or
            (dir_l and snake.is_collision(point_u))or
            (dir_r and snake.is_collision(point_d)),

            #danger in moving left
            (dir_r and snake.is_collision(point_u))or
            (dir_l and snake.is_collision(point_d))or
            (dir_u and snake.is_collision(point_l))or
            (dir_d and snake.is_collision(point_r)),

            #direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            #food location
            snake.food.x <snake.head.x,
            snake.food.x >snake.head.x,
            snake.food.y <snake.head.y,
            snake.food.y >snake.head.y
        ]
        return np.array(state,dtype=int)
        

    def remember(self,state,action,reward,next_state,game_over):
        self.memory.append((state,action,reward,next_state,game_over))
        if len(self.memory) > Max_Memory:
            self.memory.popleft()
        #popleft if maximum memory is reached
    
    def train_long_memory(self):
        if len(self.memory)>Batch_Size:
            mini_sample = random.sample(self.memory,Batch_Size)
        else:
            mini_sample=self.memory

        states = []
        actions = []
        rewards = []
        next_states = []
        game_overs = []

        for state, action, reward, next_state, game_over in mini_sample:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            game_overs.append(game_over)

        self.trainer.train_step(states, actions, rewards, next_states, game_overs)

        #states,actions,rewardss,next_states,game_overs = zip(*mini_sample)
        #self.trainer.train_step(states,actions,rewardss,next_states,game_overs)
            
        

    def train_short_memory(self,state,action,rewards,next_state,game_over):
        self.trainer.train_step(state,action,rewards,next_state,game_over)

    def get_action(self,state):
        #random moves
        self.epsilon = 100 - self.num_games
        final_move = [0,0,0]
        if random.randint(0,200)<self.epsilon:
            move = random.randint(0,2)
            final_move[move] =1
        else:
            state0 = torch.tensor(state,dtype=torch.float)
            prediction = self.model(state0)   
            move = torch.argmax(prediction).item()
            final_move[move]=1
        return final_move

def train():
        plot_scores = []
        plot_mean_scores = [] 
        total_score = 0
        record = 0
        agent = Agent()
        snake = SnakeGameAI()
        while True:
            #get current state
            old_state = agent.get_state(snake)

            #get the moves
            final_move = agent.get_action(old_state)

            #perform the move and get new state
            reward , game_over,score = snake.play_step(final_move)
            new_state = agent.get_state(snake)

            #short memory training 
            agent.train_short_memory(old_state,final_move,reward,new_state,game_over)

            #remeber
            agent.remember(old_state,final_move,reward,new_state,game_over)

            if game_over:
                #Long train memory(all previous trains)
                snake.reset()
                agent.num_games +=1
                agent.train_long_memory()

                if score>record:
                    record=score
                    agent.model.save()

                print('Game',agent.num_games,' Score',score,'Record',record)

                plot_scores.append(score)
                total_score+=score
                mean_score = total_score / agent.num_games
                plot_mean_scores.append(mean_score)
                plot(plot_scores,plot_mean_scores)

if __name__ == '__main__':
    train()
