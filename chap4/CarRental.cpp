
/*
#######################################################################
# Copyright (C)                                                       #
# 2018 Donghai He(gsutilml@gmail.com)                  #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

#######################################################################
# Copyright (C)                                                       #
# 2016 Shangtong Zhang(zhangshangtong.cpp@gmail.com)                  #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# 2017 Aja Rangaswamy (aja004@gmail.com)                              #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
*/

#include <cmath>
#include <vector>
using namespace std;

// max # of cars at each location
const int MAX_CARS = 20;
// max # of cars to move during night
const int MAX_MOVE_OF_CARS = 5;
// expectation of rental requests at first location
const int RENTAL_REQUEST_FIRST_LOC  = 3;
// expectation of rental requests at second location
const int RENTAL_REQUEST_SECOND_LOC = 4;
// expectation of # of cars returned at first location
const int RETURNS_FIRST_LOC = 3;
// expectation of # of cars returned at second location
const int RETURNS_SECOND_LOC = 2;

const double DISCOUNT = 0.9;

// credit earned by one car
const double RENTAL_CREDIT = 10;
// cost of moving one car
const double MOVE_CAR_COST = 2;
// current policy
vector<vector<double>> policy(MAX_CARS+1, vector<double>(MAX_CARS+1, 0));
// current state value
vector<vector<double>> state_values(MAX_CARS+1, vector<double>(MAX_CARS+1, 0));
// all possible states
vector<pair<int,int>> states;
// all possible actions
vector<int> actions(2*MAX_MOVE_OF_CARS+1,0);

void init_states_actions()
{
    for (int i=0;i<MAX_CARS+1;i++){
        for (int j=0;j<MAX_CARS+1;j++){
            states.push_back(make_pair(i,j));
        }
    }
    for (int i=-MAX_MOVE_OF_CARS;i<MAX_MOVE_OF_CARS+1;i++)
        actions[i+MAX_MOVE_OF_CARS] = i;
}

// @state: [# of cars at 1st location, # of cars at 2nd location]
// @action: positive when moving cars from 1st to 2nd location
//          negative when moving cars from 2nd to 1st location
// @state_vals: state value matrix
double expected_return(pair<int,int>& state, int action, vector<vector<double>>& state_vals)
{
    double total_return = -MOVE_CAR_COST * abs(action);
    //TODO:
}
